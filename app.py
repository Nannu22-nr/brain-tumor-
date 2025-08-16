from flask import Flask, render_template, request, send_from_directory
import onnxruntime as ort
from PIL import Image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the ONNX model
model_path = "models/mobilenetv2_classifier.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define uploads folder
UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Helper function to preprocess and predict
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: img_array})[0]

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == "notumor":
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_location = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)

            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence*100:.2f}%",
                file_path=f"/uploads/{file.filename}"
            )

    return render_template("index.html", result=None)

@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
