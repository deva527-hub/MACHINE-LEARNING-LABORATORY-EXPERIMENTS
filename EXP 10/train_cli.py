"""
MedVision-ML: Command-Line Interface (CLI)
Allows users to compile, train, evaluate, and test medical image diagnosis
directly from the terminal or PowerShell.

Usage:
    python train_cli.py --generate-data
    python train_cli.py --train --algo svm
    python train_cli.py --train --algo random_forest
    python train_cli.py --evaluate
    python train_cli.py --predict path/to/image.jpg
"""

import os
import argparse
import json
from PIL import Image
import ml_engine
import dataset_generator

def print_banner():
    banner = r"""
  ==============================================================
    __  ___         l _    ___      _               __  _____ 
   /  |/  /___  ___/ / |  / (_)____(_)___  ____    /  |/  / / 
  / /|_/ / _ \/ _  / | | / / / ___/ / __ \/ __ \  / /|_/ / /  
 / /  / /  __/ /_/ /  | |/ / (__  ) / /_/ / / / / / /  / / /___
/_/  /_/\___/\__,_/   |___/_/____/_/\____/_/ /_/ /_/  /_/_____/
     Medical Image Diagnosis & Machine Learning Pipeline
  ==============================================================
"""
    print(banner)

def main():
    parser = argparse.ArgumentParser(description="MedVision-ML Medical Image Classifier CLI")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic sample medical X-rays")
    parser.add_argument("--train", action="store_true", help="Train machine learning model on dataset")
    parser.add_argument("--algo", type=str, default="svm", choices=["svm", "random_forest", "knn", "logistic_regression"], help="ML Algorithm to use")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio (e.g., 0.25)")
    parser.add_argument("--evaluate", action="store_true", help="Run performance evaluation on trained model")
    parser.add_argument("--predict", type=str, help="Path to medical image file for inference")

    args = parser.parse_args()
    print_banner()

    if args.generate_data:
        print("[*] Generating clinical sample dataset...")
        dataset_generator.generate_sample_dataset()
        print("[+] Sample data successfully generated in 'static/samples/'.\n")

    if args.train:
        print(f"[*] Training {args.algo.upper()} classifier...")
        print(f"[*] Train/Test Split: {int((1-args.test_size)*100)}% Train / {int(args.test_size*100)}% Test")
        report = ml_engine.train_model(algorithm=args.algo, test_size=args.test_size)
        
        print("\n" + "="*50)
        print("           MODEL TRAINING PERFORMANCE REPORT")
        print("="*50)
        print(f"  Algorithm           : {report['algorithm']}")
        print(f"  Total Samples       : {report['total_samples']}")
        print(f"  Training Set Size   : {report['train_samples']}")
        print(f"  Testing Set Size    : {report['test_samples']}")
        print("-" * 50)
        print(f"  Accuracy            : {report['metrics']['accuracy']}%")
        print(f"  Sensitivity (Recall): {report['metrics']['sensitivity_recall']}% (True Positive Rate)")
        print(f"  Specificity         : {report['metrics']['specificity']}% (True Negative Rate)")
        print(f"  Precision           : {report['metrics']['precision']}%")
        print(f"  F1-Score            : {report['metrics']['f1_score']}%")
        print(f"  ROC-AUC Score       : {report['metrics']['roc_auc']}")
        print("-" * 50)
        cm = report['confusion_matrix']
        print(f"  Confusion Matrix    : TP={cm['tp']}  |  FP={cm['fp']}")
        print(f"                        FN={cm['fn']}  |  TN={cm['tn']}")
        print("="*50)
        print(f"[+] Model and Scaler serialized to '{ml_engine.MODELS_DIR}'.\n")

    if args.predict:
        if not os.path.exists(args.predict):
            print(f"[-] Error: Image path '{args.predict}' not found.")
            return

        print(f"[*] Loading and diagnosing image: {args.predict}")
        img = Image.open(args.predict)
        res = ml_engine.predict_medical_image(img)

        print("\n" + "="*50)
        print("             CLINICAL DIAGNOSIS RESULT")
        print("="*50)
        print(f"  Diagnostic Output   : {res['prediction']}")
        print(f"  Confidence Level    : {res['confidence']}%")
        print(f"  Risk Categorization : {res['risk_level']}")
        print("-" * 50)
        print(f"  Probabilities       : Normal: {res['probabilities']['NORMAL']}% | Pneumonia: {res['probabilities']['PNEUMONIA']}%")
        print(f"  Radiological Finding: {res['radiological_finding']}")
        print(f"  Clinical Action     : {res['recommendation']}")
        print("="*50 + "\n")

    if not any([args.generate_data, args.train, args.predict, args.evaluate]):
        parser.print_help()

if __name__ == "__main__":
    main()

