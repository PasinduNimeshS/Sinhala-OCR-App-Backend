from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uuid
from pathlib import Path
from .ocr_runner import run_ocr
import uvicorn

app = FastAPI(title="Sinhala OCR API", version="1.0")

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(400, "Only PNG/JPG/JPEG allowed")

    file_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix
    temp_path = UPLOAD_DIR / f"{file_id}{suffix}"

    try:
        contents = await file.read()
        temp_path.write_bytes(contents)

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)