"""
MedVision-ML: Flask Web Application Backend
Serves the unified single-frame dashboard and provides REST API endpoints
for the entire 5-stage medical machine learning pipeline.
"""

import os
import io
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import ml_engine
import dataset_generator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure sample dataset exists on startup
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")
if not os.path.exists(SAMPLES_DIR) or len(os.listdir(SAMPLES_DIR)) == 0:
    print("[*] Initializing default medical image samples...")
    dataset_generator.generate_sample_dataset()

# Cache currently loaded image in memory for user session
current_session = {
    "image": None,
    "metadata": None,
    "preprocessed": None,
    "features": None,
    "last_report": None
}

# Auto-train default model on startup if not present
if not os.path.exists(ml_engine.MODEL_PATH):
    print("[*] Training baseline clinical model (SVM)...")
    try:
        current_session["last_report"] = ml_engine.train_model(algorithm="svm")
    except Exception as e:
        print(f"[!] Note on initial training: {e}")

@app.route("/")
def index():
    """Render the single-frame unified clinical ML application."""
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return list of available clinical benchmark sample images."""
    samples = {"normal": [], "pneumonia": []}
    for category in ["normal", "pneumonia"]:
        cat_path = os.path.join(SAMPLES_DIR, category)
        if os.path.exists(cat_path):
            files = [f for f in sorted(os.listdir(cat_path)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            samples[category] = files[:10]  # top 10
    return jsonify({"status": "success", "samples": samples})

@app.route("/api/load-sample", methods=["POST"])
def load_sample():
    """Load a sample medical image by filename."""
    data = request.get_json() or {}
    category = data.get("category", "normal")
    filename = data.get("filename", "normal_case_01.jpg")
    
    file_path = os.path.join(SAMPLES_DIR, category, filename)
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"Sample file not found: {filename}"}), 404

    img, metadata = ml_engine.load_and_inspect_image(file_path)
    current_session["image"] = img
    current_session["metadata"] = metadata
    current_session["preprocessed"] = None
    current_session["features"] = None

    return jsonify({
        "status": "success",
        "filename": filename,
        "category": category,
        "metadata": metadata
    })

@app.route("/api/upload-image", methods=["POST"])
def upload_image():
    """Handle custom medical image upload."""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        img, metadata = ml_engine.load_and_inspect_image(pil_img)
        
        current_session["image"] = img
        current_session["metadata"] = metadata
        current_session["preprocessed"] = None
        current_session["features"] = None

        return jsonify({
            "status": "success",
            "filename": file.filename,
            "metadata": metadata
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/preprocess", methods=["POST"])
def preprocess_image():
    """Run preprocessing pipeline on current image."""
    if current_session["image"] is None:
        return jsonify({"status": "error", "message": "No medical image currently loaded. Please load or upload an image first."}), 400

    try:
        prep_result = ml_engine.run_preprocessing_pipeline(current_session["image"])
        current_session["preprocessed"] = prep_result["processed_array"]
        return jsonify({
            "status": "success",
            "stages": prep_result["stages"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/extract-features", methods=["POST"])
def extract_features():
    """Extract texture, statistical, and edge features."""
    if current_session["image"] is None:
        return jsonify({"status": "error", "message": "Please load an image first."}), 400

    try:
        if current_session["preprocessed"] is None:
            prep_result = ml_engine.run_preprocessing_pipeline(current_session["image"])
            current_session["preprocessed"] = prep_result["processed_array"]

        features = ml_engine.extract_all_features(current_session["preprocessed"])
        current_session["features"] = features

        return jsonify({
            "status": "success",
            "feature_table": features["feature_table"],
            "feature_names": features["feature_names"],
            "feature_values": [f["value"] for f in features["feature_table"]]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def train_model_endpoint():
    """Train selected ML algorithm on clinical dataset."""
    data = request.get_json() or {}
    algorithm = data.get("algorithm", "svm")
    test_size = float(data.get("test_size", 0.25))

    try:
        report = ml_engine.train_model(algorithm=algorithm, test_size=test_size)
        current_session["last_report"] = report
        return jsonify({
            "status": "success",
            "report": report
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/diagnose", methods=["POST"])
def diagnose_image():
    """Run inference on currently loaded image using the trained model."""
    if current_session["image"] is None:
        return jsonify({"status": "error", "message": "No image loaded to diagnose."}), 400

    try:
        result = ml_engine.predict_medical_image(current_session["image"])
        return jsonify({
            "status": "success",
            "result": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/run-all", methods=["POST"])
def run_all_pipeline():
    """
    Execute entire pipeline in 1 click:
    Input Image -> Preprocessing -> Feature Extraction -> Diagnosis & Evaluation
    """
    if current_session["image"] is None:
        # Load default sample if none is selected
        sample_path = os.path.join(SAMPLES_DIR, "pneumonia", "pneumonia_case_01.jpg")
        if os.path.exists(sample_path):
            img, metadata = ml_engine.load_and_inspect_image(sample_path)
            current_session["image"] = img
            current_session["metadata"] = metadata
        else:
            return jsonify({"status": "error", "message": "No image loaded and no samples available."}), 400

    try:
        # 1. Metadata
        metadata = current_session["metadata"]

        # 2. Preprocess
        prep_result = ml_engine.run_preprocessing_pipeline(current_session["image"])
        current_session["preprocessed"] = prep_result["processed_array"]

        # 3. Extract Features
        features = ml_engine.extract_all_features(prep_result["processed_array"])
        current_session["features"] = features

        # 4. Predict
        diagnosis = ml_engine.predict_medical_image(current_session["image"])

        # 5. Model Evaluation Report
        if current_session["last_report"] is None:
            current_session["last_report"] = ml_engine.train_model(algorithm="svm")

        return jsonify({
            "status": "success",
            "metadata": metadata,
            "preprocessing": prep_result["stages"],
            "features": features,
            "diagnosis": diagnosis,
            "evaluation": current_session["last_report"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    print("\n=======================================================")
    print(" MedVision-ML Web Application Server Starting...")
    print(" Local Web URL: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False)

