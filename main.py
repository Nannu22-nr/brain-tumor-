from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import shutil

# Initialize FastAPI app
app = FastAPI()

# Paths
MODEL_PATH = "models/vgg16_classifier.onnx"   # ✅ ONNX model
UPLOAD_FOLDER = "uploads"
TEMPLATES_FOLDER = "templates"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ONNX model
try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
except Exception as e:
    raise RuntimeError(f"❌ Could not load ONNX model from {MODEL_PATH}: {e}")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Mount static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)


# ---- Prediction helper ----
def predict_tumor(image_path: str):
    IMAGE_SIZE = 128
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: img_array})[0]

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = float(np.max(predictions, axis=1)[0])

    if class_labels[predicted_class_index] == "notumor":
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score


# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
async def post_index(request: Request, file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict
    result, confidence = predict_tumor(file_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "confidence": f"{confidence * 100:.2f}%",
        "file_path": f"/uploads/{file.filename}"
    })


@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return FileResponse(path=file_path, filename=filename)
