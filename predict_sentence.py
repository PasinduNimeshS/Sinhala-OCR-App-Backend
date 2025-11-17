import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import binary_dilation
import torch
import torch.nn as nn
from torchvision import models, transforms
import warnings
import shutil
warnings.filterwarnings('ignore')
import unicodedata

# ==============================
# === CONFIGURATION (shared) ===
# ==============================
IMG_PATH = ""

WORD_CROP_DIR =os.getenv("WORD_CROP_DIR", "word_crops_80x80")
os.makedirs(WORD_CROP_DIR, exist_ok=True)
# CLEAR OLD CROPS BEFORE SAVING NEW ONES
for file in os.listdir(WORD_CROP_DIR):
    file_path = os.path.join(WORD_CROP_DIR, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
print(f"Cleared old crops from '{WORD_CROP_DIR}'")

# ---- tiny internal hole filling (only black specks inside letters) ----
HOLE_FILL_K  = (5, 5)
HOLE_FILL_IT = 1

# ---- vertical gap closing (merge vertically split parts) ----
VERT_CLOSE_K  = (1, 12)
VERT_CLOSE_IT = 1

# ---- noise / dot removal ----
MIN_LETTER_AREA = 230
MIN_LETTER_W    = 15
MIN_LETTER_H    = 23

# ---- word grouping (horizontal only) ----
WORD_DILATE_W = 35
ROW_Y_THRESHOLD = 70

# ---- stroke thinning before saving ----
THIN_KERNEL = (3, 3)
THIN_ITER    = 1

# ---- 80x80 black canvas ----
CANVAS_SIZE = 80

# ==============================
# === PART 1: letters_segmented_fixed.py LOGIC ===
# ==============================

def clean_binary_internal_fill(gray):
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 12)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, open_k, iterations=1)
    fill_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, HOLE_FILL_K)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, fill_k, iterations=HOLE_FILL_IT)
    return thr


def close_vertical_gaps_only(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, VERT_CLOSE_K)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=VERT_CLOSE_IT)


def get_letter_boxes(binary):
    closed = close_vertical_gaps_only(binary)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if (area >= MIN_LETTER_AREA and w >= MIN_LETTER_W and h >= MIN_LETTER_H):
            boxes.append((x, y, w, h, area))
    return boxes


def group_into_rows(boxes):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[1])
    rows = []
    cur = [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - cur[-1][1]) < ROW_Y_THRESHOLD:
            cur.append(b)
        else:
            rows.append(sorted(cur, key=lambda b: b[0]))
            cur = [b]
    rows.append(sorted(cur, key=lambda b: b[0]))
    return rows


def get_word_boxes_from_letters(letter_boxes_per_row):
    word_boxes = []
    for row in letter_boxes_per_row:
        if not row:
            continue
        row = sorted(row, key=lambda b: b[0])
        current_word = [row[0]]
        gap_threshold = WORD_DILATE_W

        for box in row[1:]:
            last_x = current_word[-1][0] + current_word[-1][2]
            curr_x = box[0]
            if curr_x - last_x <= gap_threshold:
                current_word.append(box)
            else:
                if len(current_word) >= 1:
                    xs = [b[0] for b in current_word]
                    ys = [b[1] for b in current_word]
                    ws = [b[2] for b in current_word]
                    hs = [b[3] for b in current_word]
                    x_min = min(xs)
                    y_min = min(ys)
                    x_max = max(x + w for x, w in zip(xs, ws))
                    y_max = max(y + h for y, h in zip(ys, hs))
                    word_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
                current_word = [box]

        if len(current_word) >= 1:
            xs = [b[0] for b in current_word]
            ys = [b[1] for b in current_word]
            ws = [b[2] for b in current_word]
            hs = [b[3] for b in current_word]
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(x + w for x, w in zip(xs, ws))
            y_max = max(y + h for y, h in zip(ys, hs))
            word_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return word_boxes

sinhala_classes = ["අ","ආ","ඇ","ඈ","ඉ","ඊ","උ","එ","ඒ","ඔ","ඕ", 
"ක","කා","කැ","කෑ","කි","කී","කු","කූ","ක්","ක්‍ර","ක්‍රි","ක්‍රී",
"ග","ගා","ගැ","ගෑ","ගි","ගී","ගු","ගූ","ග්","ග්‍ර","ග්‍රි", "ග්‍රී",
"ච","චා","චැ","චෑ","චි","චී","චු","චූ","ච්","ච්‍ර","ච්‍ර්","ච්‍රී", 
"ජ","ජා","ජැ","ජෑ","ජි","ජී","ජු","ජූ","ජ්","ජ්‍ර","ජ්‍රි","ජ්‍රී",
"ට","ටා","ටැ","ටෑ","ටි","ටී","ටු","ටූ","ට්","ට්‍ර","ට්‍ර්","ට්‍රි",
"ඩ","ඩා","ඩැ","ඩෑ","ඩි","ඩී","ඩු","ඩූ","ඩ්","ඩ්‍ර","ඩ්‍ර්","ඩ්‍රි",
"ණ","ණා","ණි",
"ත","තා","ති","තී","තු","තූ","ත්","ත්‍ර","ත්‍රා","ත්‍රි","ත්‍රී",
"ද ","දා","දැ","දෑ","දි","දී","දු","දූ","ද්","ද්‍ර","ද්‍රා","ද්‍රි","ද්‍රී",
"න","නා","නැ","නෑ","නි","නී","නු","නූ","න්","න්‍ර","න්‍රා","න්‍රි","න්‍රී",
"ප","පා","පැ","පෑ","පි","පී","පු","පූ","ප්","ප්‍රෝ","ප්‍ර","ප්‍රා","ප්‍රි","ප්‍රී",
"බ","බා","බැ","බෑ","බි","බී","බු","බූ","බ්","බ්‍ර","බ්‍රා","බ්‍රි","බ්‍රී",
"ම","මා","මැ","මෑ","මි","මී","මු","මූ","ම්","ම්‍ර","ම්‍රා","ම්‍රි","ම්‍රී",
"ය","යා","යැ","යෑ","යි","යී","යු","යූ","ය්",
"ර","රා","රැ","රෑ","රු","රූ","රි","රී",
"ල","ලා","ලැ","ලෑ","ලි","ලී","ලු","ලූ","ල්",
"ව","වා","වැ","වෑ","වි","වී","වු","වූ","ව්","ව්‍ර","ව්‍රා","ව්‍රැ","ව්‍රෑ",
"ශ","ශා","ශැ","ශෑ","ශි","ශී","ශු","ශූ","ශ්","ශ්‍ර","ශ්‍රා","ශ්‍රැ","ශ්‍රෑ","ශ්‍රි","ශ්‍රී",
"ෂ","ෂා","ෂැ","ෂෑ","ෂි","ෂී","ෂු","ෂූ","ෂ්",
"ස","සා","සැ","සෑ","සි","සී","සු","සූ","ස්‍ර","ස්‍රා","ස්‍රි","ස්‍රී","ස්‍",
"හ","හා","හැ","හෑ","හි","හී","හු","හූ","හ්",
"ළ","ළා","ළැ","ළෑ","ළි","ළී",
"ළූ","ළූ",
"ෆ","ෆා","ෆැ","ෆෑ","ෆි","ෆී","ෆූ","ෆූ","ෆ්‍ර","ෆ්‍රි","ෆ්‍රී","ෆ්‍රැ","ෆ්‍රෑ","ෆ්",
"ක්‍රා","ක්‍රැ","ක්‍රෑ",
"ඛ","ඛා","ඛි","ඛී","ඛ්",
"ඝ","ඝා","ඝැ","ඝෑ","ඝි","ඝී","ඝු","ඝූ","ඝ්","ඝ්‍ර","ඝ්‍රා","ඝ්‍රි","ඝ්‍රී",
"ඳ","ඳා","ඳැ","ඳෑ","ෑ","ඳි","ඳී","ඳු","ඳූ","ඳ්",
"ඟ","ඟා","ඟැ","ඟෑ","ඟි","ඟී","ඟු","ඟූ","ඟ්",
"ඬ","ැ","ඬා","ඬැ","ඬෑ","ඬි","ඬී","ඬු","ඬූ","ඬ්",
"ඹ","ඹා","ඹැ","ඹෑ","ඹි","ඹී","ඹු","ඹූ","ඹ්",
"භ","භා","භැ","භෑ","භි","භී","භු","භූ","භ්",
"ධ","ධා","ධැ","ධෑ","ධි","ධී","ධු","ධූ","ධ්",
"ඨ","ඨා","ඨැ","ඨි","ඨී","ඨු","ඨූ","ඨ්",
"ඪ","ඪා","ඪි",
"ඵ","ඵා","ඵු","ඵි","ඵ්",
"ථ","ථා","ථැ","ථ්",
"ා","ෟ",
"ණැ","ණෑ","ෘ","ණී","ණු","ණූ","ණ්",
"ඥ","ඥා",
"ඤ","ඤා","ඤු","ඤ්",
"ඣ","ඣා","ඣු","ඣ්",
"ඦ","ඦා","ඦැ","ඦෑ","ඦි","ඦු","ඦූ","ඦෝ","ඦ්",
"ඡ","ඡා","ඡැ","ඡෑ","ඡි","ඡේ",
"තැ","තෑ","ත්‍රැ","ත්‍රෑ", 
"ළු","ෲ","ෛ ","ෙ" 
]
num_classes = len(sinhala_classes)
print(f"Number of classes: {num_classes}")

class SinhalaCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "best_sinhala_ocr_model.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SinhalaCNN(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded from: {model_path}")

def denoise_and_resize(img_array, target_size=(243, 243)):
    if len(img_array.shape) == 3:
        img = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        img = img_array.astype(np.uint8)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    img_denoised = cv2.GaussianBlur(img_resized, (3, 3), sigmaX=1)
    img_denoised = cv2.medianBlur(img_denoised, 3)
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    return cv2.filter2D(img_denoised, -1, sharpen_kernel)

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_single_image(model, img_array, transform, class_names, device):
    model.eval()
    img_denoised = denoise_and_resize(img_array)
    img_pil = Image.fromarray(img_denoised)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        _, pred = torch.max(output, 1)
        confidence = probs[pred.item()].item()
    return class_names[pred.item()], confidence

# ------------------------------------------------------------------
# Core word-recognition function (returns the word to be used in the sentence)
# ------------------------------------------------------------------
def recognize_word_image(image_path, segment_dir='temp_segments', target_size=80):
    os.makedirs(segment_dir, exist_ok=True)
    img = plt.imread(image_path)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)

    gray = np.mean(img[:, :, :3], axis=2) if len(img.shape) == 3 else img
    binary = gray > 0.5
    rows, cols = np.nonzero(binary)
    if len(rows) == 0:
        return "", "", []

    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    proj = np.sum(binary.astype(float), axis=0)
    is_nonzero = proj > 0
    concat = np.concatenate(([False], is_nonzero, [False]))
    diff = np.diff(concat.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    num_chars = len(starts)
    predicted_letters = []
    letter_confidences = []

    for i in range(num_chars):
        s_col = starts[i]
        e_col = ends[i] - 1
        sub_binary = binary[:, s_col:e_col + 1]
        sub_rows, sub_cols = np.nonzero(sub_binary)
        if len(sub_rows) == 0:
            continue
        sub_min_row = sub_rows.min()
        sub_max_row = sub_rows.max()
        sub_min_col = s_col + sub_cols.min()
        sub_max_col = s_col + sub_cols.max()
        w = sub_max_col - sub_min_col + 1
        h = sub_max_row - sub_min_row + 1

        left, top = sub_min_col, sub_min_row
        crop = img[top:top + h, left:left + w, :3] if len(img.shape) == 3 else img[top:top + h, left:left + w]
        crop_gray = np.mean(crop, axis=2) if len(crop.shape) == 3 else crop
        crop_binary = crop_gray > 0.5

        thickened = binary_dilation(crop_binary, iterations=1)

        padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
        pad_y = (target_size - h) // 2
        pad_x = (target_size - w) // 2
        padded[pad_y:pad_y + h, pad_x:pad_x + w] = np.tile(thickened.astype(np.float32)[:, :, None], (1, 1, 3))

        padded_uint8 = (padded * 255).astype(np.uint8) if padded.max() <= 1.0 else padded.astype(np.uint8)

        pred, conf = predict_single_image(model, padded_uint8, val_test_transform, sinhala_classes, device)
        predicted_letters.append(pred)
        letter_confidences.append((pred, conf))

    raw_word = ''.join(predicted_letters)

    # ------------------- Dictionary validation -------------------
    word_file_path = os.path.join(BASE_DIR, "models", "split_sinhala_words_new.txt")
    seen_groups = {
        'seen1': {'එ', 'ඵ', 'ළු', 'ව', 'ඪ', 'ථ'},
        'seen2': {'ඔ', 'ඬ', 'ඹ','ඩ','ම','ධ'},
        'seen3': {'ක', 'ත', 'න'},
        'seen4': {'හ', 'ශ', 'ග', 'භ'},
        'seen5': {'ජ', 'ඦ', 'ඡ'},
        'seen6': {'බ', 'ඛ'},
        'seen7': {'ය', 'ස', 'ඝ'}
    }
    def get_seen_group(l):
        for g in seen_groups.values():
            if l in g:
                return g
        return None

    search_nfc = unicodedata.normalize('NFC', raw_word)
    try:
        with open(word_file_path, 'r', encoding='utf-8') as f:
            raw_words = [line.strip() for line in f if line.strip()]
            word_set = {unicodedata.normalize('NFC', w) for w in raw_words}
    except FileNotFoundError:
        print(f"Warning: Dictionary {word_file_path} not found. Using raw OCR.")
        return raw_word, raw_word, letter_confidences

    if search_nfc in word_set:
        return search_nfc, raw_word, letter_confidences

    # ----- similarity search (NFC) -----
    length = len(search_nfc)
    threshold = 0.5
    best_word = None
    best_matches = 0
    for w in word_set:
        if len(w) != length:
            continue
        matches = 0
        valid = True
        for i in range(length):
            if search_nfc[i] == w[i]:
                matches += 1
            else:
                grp = get_seen_group(search_nfc[i])
                if grp is None or w[i] not in grp:
                    valid = False
                    break
        if not valid:
            continue
        sim = matches / length
        if sim >= threshold and matches > best_matches:
            best_matches = matches
            best_word = w

    if best_word:
        return best_word, raw_word, letter_confidences

    # ----- fallback NFD -----
    search_nfd = unicodedata.normalize('NFD', raw_word)
    word_set_nfd = {unicodedata.normalize('NFD', w) for w in raw_words}
    best_word = None
    best_matches = 0
    for w in word_set_nfd:
        if len(w) != length:
            continue
        matches = 0
        valid = True
        for i in range(length):
            if search_nfd[i] == w[i]:
                matches += 1
            else:
                grp = get_seen_group(search_nfd[i])
                if grp is None or w[i] not in grp:
                    valid = False
                    break
        if not valid:
            continue
        sim = matches / length
        if sim >= threshold and matches > best_matches:
            best_matches = matches
            best_word = w

    if best_word:
        return best_word, raw_word, letter_confidences

    if os.path.exists(segment_dir):
        shutil.rmtree(segment_dir)
    return raw_word, raw_word, letter_confidences


# ==============================
# === MAIN OCR PIPELINE (called by the API) ===
# ==============================
def run_ocr_pipeline():
    global IMG_PATH
    if not IMG_PATH or not os.path.isfile(IMG_PATH):
        raise ValueError("IMG_PATH not set or invalid.")

    # ---- read image -------------------------------------------------
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read {IMG_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = clean_binary_internal_fill(gray)

    # ---- letter / word detection ------------------------------------
    letter_boxes = get_letter_boxes(binary)
    letter_rows = group_into_rows(letter_boxes)
    word_boxes = get_word_boxes_from_letters(letter_rows)

    # ---- draw debug image -------------------------------------------
    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    letter_id = 1
    for row in letter_rows:
        for (x, y, w, h, area) in row:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, str(letter_id),
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            letter_id += 1
    for (x, y, w, h) in word_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite("letters_word_fixed_perfect.jpg", out)
    print(f"Detected {letter_id-1} letters. Word boxes fixed.")

    # ---- save 80x80 word crops --------------------------------------
    print(f"Saving 80x80 black canvas word crops to '{WORD_CROP_DIR}/'...")
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, THIN_KERNEL)

    for idx, (x, y, w, h) in enumerate(word_boxes):
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(binary.shape[1], x + w + pad)
        y2 = min(binary.shape[0], y + h + pad)

        word_crop = binary[y1:y2, x1:x2].copy()
        word_crop_thin = cv2.erode(word_crop, thin_kernel, iterations=THIN_ITER)

        crop_h, crop_w = word_crop_thin.shape
        scale_w = CANVAS_SIZE / crop_w
        scale_h = CANVAS_SIZE / crop_h
        scale = min(scale_w, scale_h, 1.0)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)

        if scale < 1.0:
            word_resized = cv2.resize(word_crop_thin, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            word_resized = word_crop_thin
            new_w, new_h = crop_w, crop_h

        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        y_offset = (CANVAS_SIZE - new_h) // 2
        x_offset = (CANVAS_SIZE - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = word_resized

        crop_path = os.path.join(WORD_CROP_DIR, f"word_{idx + 1}.png")
        cv2.imwrite(crop_path, canvas)
        print(f"   → Saved {os.path.basename(crop_path)} ({crop_w}×{crop_h} → {new_w}×{new_h} centred)")

    # ---- recognise each word ----------------------------------------
    print("\n" + "="*60)
    print("STARTING WORD RECOGNITION ON ALL CROPS")
    print("="*60)

    word_crop_files = sorted(
        [f for f in os.listdir(WORD_CROP_DIR) if f.lower().endswith('.png')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    sentence_parts = []
    for word_file in word_crop_files:
        word_path = os.path.join(WORD_CROP_DIR, word_file)
        print(f"\n--- Processing {word_file} ---")
        final_word, raw_word, _ = recognize_word_image(word_path)
        sentence_parts.append(final_word)
        print(f"Raw OCR: {raw_word} → Final word: {final_word}")

    final_sentence = " ".join(sentence_parts)

    print("\n" + "="*60)
    print("FINAL PREDICTED SENTENCE")
    print("="*60)
    print(final_sentence)
    print("="*60)

    with open("predicted_sentence.txt", "w", encoding="utf-8") as f:
        f.write(final_sentence)
    print("Sentence also saved to 'predicted_sentence.txt'")

    return final_sentence