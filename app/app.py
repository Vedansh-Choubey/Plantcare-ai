import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename   # for security

app = Flask(__name__)

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload folder inside app/static/uploads (works on Render + locally)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model with correct path (no more "../model/..." error)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "plant_disease_model.h5")
model = load_model(MODEL_PATH, compile=False)
print(f"✅ Model loaded successfully from: {MODEL_PATH}")

# Load class names from file (created once locally)
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "..", "model", "class_names.txt")
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]
print(f"✅ Loaded {len(class_names)} class names")

# ===================== PREDICTION FUNCTION =====================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    return class_names[index]


# ===================== ROUTES =====================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Secure filename + save
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # URL path for HTML template (so <img src="{{ image }}"> works)
    image_url = f"static/uploads/{filename}"

    # Predict
    result = predict_image(save_path)

    plant = result.split("___")[0]

    return render_template(
        "result.html",
        image=image_url,
        plant=plant,
        disease=result
    )


# ===================== RUN (for local testing) =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)