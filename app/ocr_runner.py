import os
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))

import predict_sentence as ps

def run_ocr(uploaded_image_path: str) -> str:
    print(f"[DEBUG] OCR running on: {uploaded_image_path}")

    # --- ADD: Verify image exists and is readable ---
    if not os.path.exists(uploaded_image_path):
        raise FileNotFoundError(f"Image not found: {uploaded_image_path}")
    
    test = cv2.imread(uploaded_image_path)
    if test is None:
        raise RuntimeError(f"OpenCV failed to read image: {uploaded_image_path}")

    ps.IMG_PATH = uploaded_image_path
    word_dir = PROJECT_ROOT / "word_crops_80x80"
    if word_dir.exists():
        import shutil
        shutil.rmtree(word_dir)
    word_dir.mkdir(exist_ok=True)

    final_sentence = ps.run_ocr_pipeline()
    return final_sentence