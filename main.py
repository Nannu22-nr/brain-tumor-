from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import shutil

# Initialize FastAPI app
app = FastAPI()

# Paths
MODEL_PATH = "models/vgg16_classifier.h5"
UPLOAD_FOLDER = "uploads"
TEMPLATES_FOLDER = "templates"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model safely
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model from {MODEL_PATH}: {e}")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Mount static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
templates = Jinja2Templates(directory=TEMPLATES_FOLDER)


# ---- Prediction helper ----
def predict_tumor(image_path: str):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
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
