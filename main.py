from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import shutil

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = load_model('models/vgg16_classifier.h5')

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
templates = Jinja2Templates(directory="templates")

# Prediction helper
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Main route
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# POST route for file upload and prediction
@app.post("/", response_class=HTMLResponse)
async def post_index(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result, confidence = predict_tumor(file_path)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "confidence": f"{confidence * 100:.2f}%",
        "file_path": f"/uploads/{file.filename}"
    })

# Optional: download endpoint
@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return FileResponse(path=file_path, filename=filename)

