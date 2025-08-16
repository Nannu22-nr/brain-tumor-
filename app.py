from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = "models/mobilenetv2_classifier.h5"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model safely
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model from {MODEL_PATH}: {e}")

# Class labels
class_labels = ["pituitary", "glioma", "notumor", "meningioma"]

# ---- Prediction helper ----
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = float(np.max(predictions, axis=1)[0])

    if class_labels[predicted_class_index] == "notumor":
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score


# ---- Routes ----
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


# Only needed for local development
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
