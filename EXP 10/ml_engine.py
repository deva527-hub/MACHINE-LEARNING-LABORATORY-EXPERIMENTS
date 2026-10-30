"""
MedVision-ML: Machine Learning Engine for Medical Image Analysis
Covers:
1. Image Loading & Metadata Analysis
2. Step-by-Step Preprocessing Pipeline
3. Texture & Statistical Feature Extraction (GLCM, Moments, Edges)
4. Multi-algorithm ML Training (SVM, Random Forest, KNN, Logistic Regression)
5. Comprehensive Performance Evaluation (Accuracy, Recall, Specificity, Confusion Matrix, ROC-AUC)
"""

import os
import io
import base64
import warnings
import joblib
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

warnings.filterwarnings("ignore", category=FutureWarning)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "medical_classifier.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

def pil_to_base64(pil_img, format="PNG"):
    """Convert PIL image to base64 data URI string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def numpy_to_base64(np_array, format="PNG"):
    """Convert a numpy array (0-255 or 0-1) to base64 data URI string."""
    if np_array.dtype != np.uint8:
        if np_array.max() <= 1.0 and np_array.min() >= 0.0:
            arr = (np_array * 255.0).astype(np.uint8)
        else:
            arr_norm = (np_array - np_array.min()) / (np_array.max() - np_array.min() + 1e-8)
            arr = (arr_norm * 255.0).astype(np.uint8)
    else:
        arr = np_array
    pil_img = Image.fromarray(arr)
    return pil_to_base64(pil_img, format)

# -------------------------------------------------------------
# 1. LOAD INPUT IMAGE & METADATA
# -------------------------------------------------------------
def load_and_inspect_image(file_or_path):
    """
    Load an image from a filepath or file-like object and extract metadata.
    """
    if isinstance(file_or_path, str):
        img = Image.open(file_or_path)
    else:
        img = Image.open(file_or_path)
        
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)
    
    # Calculate pixel intensity histogram (16 bins for quick UI display)
    gray_arr = np.array(img.convert("L"))
    hist, bin_edges = np.histogram(gray_arr, bins=16, range=(0, 256))
    
    metadata = {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": img.format or "JPEG",
        "aspect_ratio": round(img.width / max(1, img.height), 2),
        "mean_intensity": round(float(np.mean(gray_arr)), 2),
        "std_intensity": round(float(np.std(gray_arr)), 2),
        "min_intensity": int(np.min(gray_arr)),
        "max_intensity": int(np.max(gray_arr)),
        "histogram_counts": [int(x) for x in hist],
        "histogram_bins": [int(x) for x in bin_edges[:-1]],
        "preview_base64": pil_to_base64(img_rgb)
    }
    return img_rgb, metadata

# -------------------------------------------------------------
# 2. PREPROCESSING PIPELINE
# -------------------------------------------------------------
def run_preprocessing_pipeline(pil_img, target_size=(128, 128)):
    """
    Execute step-by-step clinical image preprocessing pipeline:
    1. Grayscale Conversion
    2. Spatial Resizing
    3. Noise Filtering (Denoising)
    4. Contrast Enhancement (Histogram Equalization)
    5. Normalization
    """
    stages = {}

    # Stage 0: Original
    stages["stage_0_original"] = {
        "title": "Original Medical Image",
        "description": "Raw uncalibrated radiological input image directly from sensor/PACS.",
        "preview": pil_to_base64(pil_img)
    }

    # Stage 1: Grayscale Conversion
    gray_img = pil_img.convert("L")
    stages["stage_1_grayscale"] = {
        "title": "Grayscale Luminance Conversion",
        "description": "Converts 3-channel RGB to single-channel intensity (Y = 0.299R + 0.587G + 0.114B).",
        "preview": pil_to_base64(gray_img)
    }

    # Stage 2: Spatial Resizing
    resized_img = gray_img.resize(target_size, Image.Resampling.BILINEAR)
    stages["stage_2_resized"] = {
        "title": f"Spatial Resizing ({target_size[0]}x{target_size[1]})",
        "description": f"Standardizes varying scanner resolutions into a uniform canonical dimension ({target_size[0]}x{target_size[1]}).",
        "preview": pil_to_base64(resized_img)
    }

    resized_arr = np.array(resized_img, dtype=np.float32)

    # Stage 3: Noise Filtering (Median / Gaussian Denoising)
    # Median filter preserves edge boundaries while eliminating quantum mottle noise
    denoised_arr = ndimage.median_filter(resized_arr, size=3)
    stages["stage_3_denoised"] = {
        "title": "Noise Denoising (3x3 Median Filter)",
        "description": "Suppresses high-frequency sensor noise and speckle artifacts while preserving fine vascular lung tissue borders.",
        "preview": numpy_to_base64(denoised_arr)
    }

    # Stage 4: Contrast Enhancement (Histogram Equalization)
    # Enhances soft-tissue pulmonary opacities that are otherwise faint
    denoised_uint8 = np.clip(denoised_arr, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(denoised_uint8.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / (cdf[-1] + 1e-8)
    enhanced_arr = np.interp(denoised_uint8.flatten(), range(256), cdf_normalized).reshape(denoised_uint8.shape)
    stages["stage_4_enhanced"] = {
        "title": "Contrast Enhancement (Hist. Equalization)",
        "description": "Expands the dynamic range of pixel intensities, highlighting subtle alveolar consolidations and lung parenchyma.",
        "preview": numpy_to_base64(enhanced_arr)
    }

    # Stage 5: Intensity Normalization [0.0, 1.0]
    normalized_arr = (enhanced_arr - enhanced_arr.min()) / (enhanced_arr.max() - enhanced_arr.min() + 1e-8)
    stages["stage_5_normalized"] = {
        "title": "Intensity Normalization [0.0 - 1.0]",
        "description": "Scales feature inputs to numerical stability [0, 1] range to prevent gradient skew during ML optimization.",
        "preview": numpy_to_base64(normalized_arr)
    }

    return {
        "stages": stages,
        "processed_array": normalized_arr  # Final 2D numpy array [0.0, 1.0]
    }

# -------------------------------------------------------------
# 3. FEATURE EXTRACTION (GLCM, Statistical Moments, Edges)
# -------------------------------------------------------------
def compute_glcm(image_norm, levels=16, dx=1, dy=0):
    """
    Compute Gray-Level Co-occurrence Matrix (GLCM).
    image_norm: 2D array normalized in [0, 1].
    """
    quantized = np.clip((image_norm * (levels - 1)).astype(np.int32), 0, levels - 1)
    h, w = quantized.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    # Co-occurrence coordinates
    y_idx = np.arange(0, h - abs(dy))
    x_idx = np.arange(0, w - abs(dx))
    
    if dy >= 0 and dx >= 0:
        row_orig = quantized[y_idx[:, None], x_idx]
        col_pair = quantized[y_idx[:, None] + dy, x_idx + dx]
    else:
        # Generic slice
        row_orig = quantized[:h-1, :w-1]
        col_pair = quantized[1:, 1:]

    for i in range(levels):
        for j in range(levels):
            glcm[i, j] = np.sum((row_orig == i) & (col_pair == j))

    # Normalize GLCM to probability distribution
    total = np.sum(glcm)
    if total > 0:
        glcm /= total
    return glcm

def extract_glcm_features(glcm):
    """
    Extract Haralick texture features from normalized GLCM.
    Contrast, Dissimilarity, Homogeneity, Energy, Correlation.
    """
    levels = glcm.shape[0]
    i_indices, j_indices = np.indices((levels, levels))

    # 1. Contrast: measures intensity variance between neighbor pixels
    contrast = np.sum(glcm * ((i_indices - j_indices) ** 2))

    # 2. Dissimilarity: linear difference
    dissimilarity = np.sum(glcm * np.abs(i_indices - j_indices))

    # 3. Homogeneity (Inverse Difference Moment): measures closeness of element distribution to diagonal
    homogeneity = np.sum(glcm / (1.0 + (i_indices - j_indices) ** 2))

    # 4. Energy (Angular Second Moment): uniformity of texture
    energy = np.sum(glcm ** 2)

    # 5. Correlation: linear dependency of gray levels
    i_mean = np.sum(i_indices * glcm)
    j_mean = np.sum(j_indices * glcm)
    i_var = np.sum(((i_indices - i_mean) ** 2) * glcm)
    j_var = np.sum(((j_indices - j_mean) ** 2) * glcm)
    i_std = np.sqrt(max(1e-8, i_var))
    j_std = np.sqrt(max(1e-8, j_var))
    
    correlation = np.sum((i_indices - i_mean) * (j_indices - j_mean) * glcm) / (i_std * j_std + 1e-8)

    return {
        "glcm_contrast": float(contrast),
        "glcm_dissimilarity": float(dissimilarity),
        "glcm_homogeneity": float(homogeneity),
        "glcm_energy": float(energy),
        "glcm_correlation": float(correlation)
    }

def extract_statistical_features(image_norm):
    """
    Extract 1st-order statistical moments:
    Mean, Std, Skewness, Kurtosis, Shannon Entropy.
    """
    flat = image_norm.flatten()
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    skew_val = float(skew(flat))
    kurt_val = float(kurtosis(flat))

    # Shannon Entropy
    hist, _ = np.histogram(flat, bins=32, range=(0.0, 1.0))
    p = hist / (np.sum(hist) + 1e-8)
    p = p[p > 0]
    entropy_val = float(-np.sum(p * np.log2(p)))

    return {
        "stat_mean": mean_val,
        "stat_std": std_val,
        "stat_skewness": skew_val,
        "stat_kurtosis": kurt_val,
        "stat_entropy": entropy_val
    }

def extract_edge_features(image_norm):
    """
    Extract gradient and edge density using Sobel & Laplacian operators.
    """
    # Sobel Horizontal and Vertical
    sx = ndimage.sobel(image_norm, axis=0)
    sy = ndimage.sobel(image_norm, axis=1)
    sobel_mag = np.hypot(sx, sy)
    
    # Laplacian (2nd derivative)
    laplacian = ndimage.laplace(image_norm)

    return {
        "edge_sobel_mean": float(np.mean(sobel_mag)),
        "edge_sobel_std": float(np.std(sobel_mag)),
        "edge_laplacian_var": float(np.var(laplacian))
    }

def extract_all_features(image_norm):
    """
    Full feature extraction pipeline combining GLCM, Stats, and Edge descriptors.
    Returns:
        dict: Human-readable named feature dictionary with descriptions
        list: Numerical feature vector for ML classifiers
    """
    glcm = compute_glcm(image_norm, levels=16, dx=1, dy=0)
    glcm_feat = extract_glcm_features(glcm)
    stat_feat = extract_statistical_features(image_norm)
    edge_feat = extract_edge_features(image_norm)

    all_feat = {**glcm_feat, **stat_feat, **edge_feat}

    # Order of features in the feature vector
    feature_names = [
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
        "glcm_energy", "glcm_correlation", "stat_mean", "stat_std",
        "stat_skewness", "stat_kurtosis", "stat_entropy",
        "edge_sobel_mean", "edge_sobel_std", "edge_laplacian_var"
    ]

    feature_descriptions = {
        "glcm_contrast": "Measures intensity variation between adjacent pixels (high in consolidated tissue)",
        "glcm_dissimilarity": "Measures localized non-uniformity across tissue borders",
        "glcm_homogeneity": "Measures overall smoothness and uniformity of lung aeration",
        "glcm_energy": "Textural orderliness (high in uniform healthy aeration)",
        "glcm_correlation": "Linear pixel dependency across gray levels",
        "stat_mean": "Average optical radiodensity across lung field",
        "stat_std": "Standard deviation of radiological intensity",
        "stat_skewness": "Asymmetry of intensity distribution (shifted by lesions)",
        "stat_kurtosis": "Peakedness of intensity histogram",
        "stat_entropy": "Shannon information complexity of tissue structure",
        "edge_sobel_mean": "Average edge gradient sharpness (ribs vs infiltrates)",
        "edge_sobel_std": "Dispersion of tissue boundary gradients",
        "edge_laplacian_var": "High-frequency curvature energy / focal sharpness"
    }

    feature_table = []
    vector = []
    for name in feature_names:
        val = all_feat[name]
        vector.append(val)
        feature_table.append({
            "name": name,
            "value": round(val, 4),
            "description": feature_descriptions.get(name, "")
        })

    return {
        "feature_dict": all_feat,
        "feature_vector": vector,
        "feature_table": feature_table,
        "feature_names": feature_names
    }

# -------------------------------------------------------------
# 4. DATASET FEATURE EXTRACTION & CACHING
# -------------------------------------------------------------
def extract_dataset_features(samples_dir="static/samples"):
    """
    Extract features from all sample images in static/samples directory.
    """
    X = []
    y = []
    filenames = []

    classes = ["normal", "pneumonia"]
    for label_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(samples_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(cls_dir, fname)
                try:
                    pil_img = Image.open(fpath)
                    prep_res = run_preprocessing_pipeline(pil_img)
                    feat_res = extract_all_features(prep_res["processed_array"])
                    X.append(feat_res["feature_vector"])
                    y.append(label_idx)
                    filenames.append(f"{cls_name}/{fname}")
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), filenames

# -------------------------------------------------------------
# 5. MODEL TRAINING & PERFORMANCE EVALUATION
# -------------------------------------------------------------
def train_model(algorithm="svm", test_size=0.25, random_state=42):
    """
    Train machine learning classifier on extracted dataset features.
    algorithm: 'svm', 'random_forest', 'knn', 'logistic_regression'
    """
    # 1. Load or extract features
    X, y, filenames = extract_dataset_features()
    
    if len(X) < 10:
        raise ValueError(f"Insufficient training samples found ({len(X)}). Please generate sample dataset first.")

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Select Classifier
    algo_key = algorithm.lower()
    if algo_key == "svm":
        model = SVC(kernel="rbf", C=1.5, probability=True, random_state=random_state)
        algo_name = "Support Vector Machine (RBF Kernel)"
    elif algo_key == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        algo_name = "Random Forest Classifier"
    elif algo_key == "knn":
        model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        algo_name = "K-Nearest Neighbors (k=5)"
    elif algo_key == "logistic_regression":
        model = LogisticRegression(max_iter=500, random_state=random_state)
        algo_name = "Logistic Regression"
    else:
        model = SVC(kernel="rbf", probability=True, random_state=random_state)
        algo_name = "Support Vector Machine (Default)"

    # 5. Fit Model
    model.fit(X_train_scaled, y_train)

    # 6. Predict on Test Set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # 7. Evaluate Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))  # Sensitivity
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    spec = float(tn / (tn + fp + 1e-8))  # Specificity

    # ROC Curve & AUC
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = float(auc(fpr, tpr))
        roc_curve_data = {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr]
        }
    else:
        roc_auc = 0.5
        roc_curve_data = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}

    # Save trained model and scaler to disk
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    evaluation_report = {
        "algorithm": algo_name,
        "algo_key": algo_key,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(X),
        "metrics": {
            "accuracy": round(acc * 100, 2),
            "sensitivity_recall": round(rec * 100, 2),
            "specificity": round(spec * 100, 2),
            "precision": round(prec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(roc_auc, 4)
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        },
        "roc_curve": roc_curve_data
    }

    return evaluation_report

# -------------------------------------------------------------
# 6. LIVE IMAGE PREDICTION / DIAGNOSIS
# -------------------------------------------------------------
def predict_medical_image(pil_img):
    """
    Run the trained model on an incoming image to provide diagnosis & confidence.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        # Auto-train default SVM model if not trained yet
        train_model(algorithm="svm")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 1. Preprocess
    prep_res = run_preprocessing_pipeline(pil_img)
    proc_arr = prep_res["processed_array"]

    # 2. Extract Features
    feat_res = extract_all_features(proc_arr)
    vector = np.array(feat_res["feature_vector"]).reshape(1, -1)

    # 3. Scale Features
    vector_scaled = scaler.transform(vector)

    # 4. Predict
    prediction_idx = int(model.predict(vector_scaled)[0])
    probabilities = model.predict_proba(vector_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    
    classes = ["NORMAL", "PNEUMONIA"]
    predicted_label = classes[prediction_idx]
    confidence = float(probabilities[prediction_idx] * 100.0)

    # Clinical interpretation note
    if predicted_label == "PNEUMONIA":
        finding = "Patchy pulmonary infiltrate / alveolar consolidation detected. Increased textural heterogeneity observed."
        risk_level = "HIGH RISK"
        recommendation = "Recommend clinical correlation, sputum culture, and follow-up high-resolution imaging."
    else:
        finding = "Clear lung fields, sharp costophrenic angles, normal radiographic parenchymal texture."
        risk_level = "LOW RISK"
        recommendation = "No active focal consolidation observed. Routine preventive care recommended."

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 2),
        "risk_level": risk_level,
        "probabilities": {
            "NORMAL": round(float(probabilities[0] * 100.0), 2),
            "PNEUMONIA": round(float(probabilities[1] * 100.0), 2)
        },
        "radiological_finding": finding,
        "recommendation": recommendation,
        "preprocessing_stages": prep_res["stages"],
        "features": feat_res["feature_table"]
    }
