# MedVision-ML: Medical Imaging Machine Learning Pipeline

An end-to-end medical image diagnostic platform that integrates **Image Loading**, **Clinical Preprocessing**, **Radiological Feature Extraction (GLCM & Statistical Moments)**, **Multi-Algorithm Model Training**, and **Performance Evaluation (Sensitivity, Specificity, Confusion Matrix, ROC-AUC)** into a single unified web application and terminal interface.

---

## 📋 Table of Contents
1. [Architecture & Pipeline Overview](#-architecture--pipeline-overview)
2. [The 5 Core Pipeline Stages](#-the-5-core-pipeline-stages)
3. [Quick Start: How to Compile & Run](#-quick-start-how-to-compile--run)
   - [Method 1: Interactive Single-Frame Web Application](#method-1-interactive-single-frame-web-application)
   - [Method 2: Command-Line Interface (CLI)](#method-2-command-line-interface-cli)
4. [Machine Learning Algorithms & Metrics](#-machine-learning-algorithms--metrics)
5. [Using Your Own Medical Dataset](#-using-your-own-medical-dataset)
6. [Project Structure](#-project-structure)

---

## 🔬 Architecture & Pipeline Overview

```
[ Medical Scan / X-Ray ]
          │
          ▼
1. Image Ingestion ───► Extract Metadata, Aspect Ratio, & Intensity Histogram
          │
          ▼
2. Preprocessor     ───► Grayscale ➔ Resizing (128x128) ➔ Denoising (Median Filter)
                         ➔ Contrast Enhancement (Equalization) ➔ Normalization [0, 1]
          │
          ▼
3. Feature Extraction ─► GLCM Haralick Textures (Contrast, Homogeneity, Energy, Correlation)
                         + 1st-Order Statistical Moments (Mean, Std, Skewness, Kurtosis, Entropy)
                         + Edge Energy (Sobel Magnitude & Laplacian)
          │
          ▼
4. Model Training   ───► Supervised Classifiers: SVM (RBF), Random Forest, KNN, Logistic Regression
                         with Stratified Cross-Validation
          │
          ▼
5. Performance &    ───► Accuracy, Sensitivity (Recall), Specificity, Precision, F1,
   Diagnosis             Confusion Matrix (TP/FP/TN/FN), ROC-AUC Curve & Clinical Verdict
```

---

## 🩺 The 5 Core Pipeline Stages

### 1. Load Input Image & Metadata
- **Input:** Supports standard radiological formats (`.jpg`, `.jpeg`, `.png`) or custom clinical scans.
- **Diagnostics:** Automatically parses image dimensions, aspect ratio, color channel modes, pixel intensity bounds, and renders an interactive **Pixel Intensity Histogram (Radiodensity distribution)**.
- **Benchmark Library:** Comes pre-loaded with simulated normal and pneumonia Chest X-Ray cases.

### 2. Preprocessor Pipeline
Medical scans contain scanner artifacts and varying contrast. The preprocessor applies a 5-step pipeline:
1. **Grayscale Luminance Conversion:** Converts RGB channels to standardized optical luminance:
   $$Y = 0.299R + 0.587G + 0.114B$$
2. **Canonical Spatial Resizing:** Standardizes arbitrary scanner resolutions into uniform $128 \times 128$ matrices using bilinear interpolation.
3. **Noise Filtering (3x3 Median Filter):** Eliminates quantum mottle sensor noise while preserving delicate anatomical boundaries (e.g., ribs and lung borders).
4. **Contrast Enhancement (Histogram Equalization):** Spreads out narrow radiodensity bands to reveal hidden soft-tissue infiltrates and consolidations.
5. **Intensity Normalization:** Scales pixel values to the range $[0.0, 1.0]$ for stable numerical optimization in gradient-based classifiers.

### 3. Feature Extraction
Extracts 13 quantitative radiological descriptors:
- **Gray-Level Co-occurrence Matrix (GLCM) Haralick Descriptors:**
  - *Contrast:* Measures local intensity variations and sharpness of consolidate borders.
  - *Dissimilarity:* Quantifies localized non-uniformity across tissue borders.
  - *Homogeneity:* Evaluates smoothness of lung aeration.
  - *Energy (Angular Second Moment):* Uniformity of tissue structure.
  - *Correlation:* Linear pixel dependency across gray levels.
- **First-Order Statistical Moments:** Mean intensity, Standard Deviation, Skewness (asymmetry), Kurtosis (peakedness), and Shannon Information Entropy ($-\sum p \log_2 p$).
- **Edge Descriptors:** Sobel gradient magnitude mean, dispersion, and Laplacian curvature variance.

### 4. Model Training
Train and compare 4 industry-standard machine learning classifiers:
- **Support Vector Machine (SVM):** Utilizes Radial Basis Function (RBF) kernel to project non-linear radiological features into high-dimensional hyperplanes.
- **Random Forest:** Ensemble of decorrelated decision trees robust to clinical noise.
- **K-Nearest Neighbors (KNN):** Distance-based instance classifier ($k=5$).
- **Logistic Regression:** Regularized linear probabilistic model.
- Includes configurable **Train/Test Split ratios** (e.g., 75% train / 25% test).

### 5. Performance Evaluation & Clinical Inference
- **Clinical Metrics:**
  - $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
  - $\text{Sensitivity (Recall)} = \frac{TP}{TP + FN}$ *(Critical in medicine: avoiding missed infections)*
  - $\text{Specificity} = \frac{TN}{TN + FP}$ *(Avoiding false alarms in healthy patients)*
  - $\text{Precision} = \frac{TP}{TP + FP}$
  - $\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
  - $\text{ROC-AUC:}$ Area Under Receiver Operating Characteristic Curve.
- **Confusion Matrix:** Interactive heatmap showing True Positives, True Negatives, False Positives (Type I Error), and False Negatives (Type II Error).
- **Live Clinical Diagnosis:** Evaluates the active image and reports:
  - Diagnosis label (`NORMAL` vs `PNEUMONIA`)
  - Confidence percentage (e.g. `97.2%`)
  - Risk classification (`LOW RISK` vs `HIGH RISK`)
  - Radiologist finding summary and recommended next clinical actions.

---

## 🚀 Quick Start: How to Compile & Run

### Step 1: Open PowerShell or Terminal
Navigate to the project directory:
```powershell
cd "c:\Users\devak\OneDrive\Desktop\ML 10"
```

### Step 2: Install Required Dependencies
All libraries are standard Python packages:
```powershell
pip install -r requirements.txt
```
*(Packages used: `Flask`, `numpy`, `scipy`, `scikit-learn`, `Pillow`, `matplotlib`, `joblib`)*

---

### Method 1: Interactive Single-Frame Web Application

1. **Start the Web Server:**
   ```powershell
   python app.py
   ```
2. **Open in Browser:**
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```
3. **Using the Application:**
   - **Load Image:** Click on any sample case (Normal or Pneumonia) or drag & drop your own scan.
   - **1-Click Full Pipeline:** Click the blue **"⚡ Run Full Pipeline"** button in the header. The app will automatically execute all 5 stages and present the live diagnosis with performance graphs.
   - **Step-by-Step Inspection:** Use the stepper tabs at the top to inspect intermediate filter outputs, feature tables, and model comparisons.

---

### Method 2: Command-Line Interface (CLI)

You can also compile, train, and test directly from the terminal using `train_cli.py`:

1. **Generate Simulated Sample Medical Images:**
   ```powershell
   python train_cli.py --generate-data
   ```

2. **Train & Evaluate SVM Classifier:**
   ```powershell
   python train_cli.py --train --algo svm --test-size 0.25
   ```

3. **Train & Evaluate Random Forest Classifier:**
   ```powershell
   python train_cli.py --train --algo random_forest
   ```

4. **Run Live Diagnosis on an Image File:**
   ```powershell
   python train_cli.py --predict static/samples/pneumonia/pneumonia_case_01.jpg
   ```
   *(Or on a normal case)*:
   ```powershell
   python train_cli.py --predict static/samples/normal/normal_case_01.jpg
   ```

---

## 📊 Machine Learning Algorithms & Metrics

| Metric | Clinical Medical Meaning | Optimal Range |
| :--- | :--- | :--- |
| **Sensitivity (Recall)** | Ability to identify diseased patients. High sensitivity guarantees zero missed pneumonias. | $> 90\%$ |
| **Specificity** | Ability to identify healthy individuals. Prevents unnecessary antibiotic treatments. | $> 85\%$ |
| **Precision** | Probability that a positive test indicates true infection. | $> 85\%$ |
| **F1-Score** | Harmonic mean of Precision and Recall. Balances both errors. | $> 85\%$ |
| **ROC-AUC** | Diagnostic discrimination ability across all potential decision thresholds. | $0.90 - 1.00$ |

---

## 📁 Using Your Own Medical Dataset

To train the model on your own real medical datasets (such as Kaggle Chest X-Ray, NIH ChestX-ray8, or Brain Tumor MRI):
1. Place your normal/healthy scans inside:
   `static/samples/normal/`
2. Place your infected/abnormal scans inside:
   `static/samples/pneumonia/`
3. Click **"Train Classifier Now"** in the web dashboard or run:
   ```powershell
   python train_cli.py --train --algo svm
   ```
The pipeline automatically extracts features from your scans and retrains the model!

---

## 📂 Project Structure

```
ML 10/
│
├── app.py                     # Flask REST API server & web application entry point
├── ml_engine.py               # Complete ML engine: Load, Preprocess, GLCM, Training, Evaluation
├── dataset_generator.py       # Realistic medical sample synthesizer & dataset loader
├── train_cli.py               # Command-line interface to train & test from terminal
├── requirements.txt           # Python dependency specifications
├── README.md                  # Complete documentation and user guide
│
├── templates/
│   └── index.html             # Unified single-frame dashboard (Tailwind + Chart.js)
│
├── static/
│   ├── css/
│   │   └── style.css          # Medical glassmorphic UI design
│   ├── js/
│   │   └── main.js            # Client-side reactivity, Chart.js graphs & REST calls
│   └── samples/
│       ├── normal/            # Control normal medical scans
│       └── pneumonia/         # Pathological medical scans
│
└── models/                    # Serialized machine learning models (.joblib)
    ├── medical_classifier.joblib
    └── scaler.joblib
```

