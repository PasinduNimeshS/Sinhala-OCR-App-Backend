# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
from pathlib import Path
from .ocr_runner import run_ocr
import cv2

app = FastAPI(title="Sinhala OCR API", version="1.0")

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(400, "Only PNG/JPG/JPEG allowed")

    file_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix
    temp_path = UPLOAD_DIR / f"{file_id}{suffix}"

    try:
        # --- FIX: Read and save image properly ---
        contents = await file.read()
        temp_path.write_bytes(contents)

        # --- ADD: Verify image can be read ---
        test_img = cv2.imread(str(temp_path))
        if test_img is None:
            raise HTTPException(500, "Uploaded image is corrupted or not readable by OpenCV")

        # --- Run OCR ---
        sentence = run_ocr(str(temp_path))

        return JSONResponse({
            "predicted_sentence": sentence,
            "image_id": file_id
        })
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()