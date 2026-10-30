"""
MedVision-ML: Medical Dataset Generator
Generates realistic synthetic Chest X-Ray medical images and feature datasets
for demonstration and training when external datasets are not yet loaded.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_chest_xray(case_type="normal", seed=None):
    """
    Synthesize a 256x256 Chest X-Ray like medical image.
    case_type: 'normal' or 'pneumonia'
    """
    if seed is not None:
        np.random.seed(seed)
        
    width, height = 256, 256
    img_array = np.zeros((height, width), dtype=np.float32)

    # 1. Background anatomy: Rib cage & thoracic cavity gradient
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2.0, height / 2.0
    
    # Elliptical thoracic boundary
    chest_mask = ((x - center_x) ** 2) / (105.0 ** 2) + ((y - center_y) ** 2) / (115.0 ** 2) <= 1.0
    img_array[chest_mask] = 160.0 + np.random.normal(0, 8, np.sum(chest_mask))

    # 2. Spine & Mediastinum (central dense white column)
    spine_mask = (np.abs(x - center_x) < 22) & (y > 20) & (y < 235)
    img_array[spine_mask] += 65.0

    # 3. Left and Right Lung Cavities (radiolucent = darker regions)
    # Left lung
    left_lung_mask = (((x - 75) ** 2) / (38 ** 2) + ((y - 125) ** 2) / (65 ** 2) <= 1.0) & (np.abs(x - center_x) >= 18)
    # Right lung
    right_lung_mask = (((x - 181) ** 2) / (38 ** 2) + ((y - 125) ** 2) / (65 ** 2) <= 1.0) & (np.abs(x - center_x) >= 18)

    # Lungs absorb fewer X-rays, so they appear darker with fine vascular markings
    img_array[left_lung_mask] = 45.0 + np.random.normal(0, 10, np.sum(left_lung_mask))
    img_array[right_lung_mask] = 45.0 + np.random.normal(0, 10, np.sum(right_lung_mask))

    # 4. Heart shadow (Cardiac silhouette on patient's left / image right)
    heart_mask = (((x - 148) ** 2) / (32 ** 2) + ((y - 150) ** 2) / (40 ** 2) <= 1.0)
    img_array[heart_mask] += 55.0

    # 5. Rib shadows (horizontal curved lighter lines)
    for rib_y in range(60, 210, 26):
        rib_curve = rib_y + 10 * np.sin((x - center_x) / 35.0)
        rib_mask = (np.abs(y - rib_curve) < 4) & chest_mask & (~spine_mask)
        img_array[rib_mask] += 30.0

    # 6. Clinical Abnormality: Infiltrates / Consolidation if Pneumonia
    if case_type == "pneumonia":
        # Introduce cloudy alveolar consolidations (bright patchy opacities in lungs)
        # Infiltrate patch 1 (mid-to-lower zone)
        patch1_x = np.random.choice([80, 175])
        patch1_y = np.random.randint(115, 170)
        r1 = np.random.randint(22, 38)
        inf1_mask = ((x - patch1_x)**2 + (y - patch1_y)**2) <= r1**2
        img_array[inf1_mask & (left_lung_mask | right_lung_mask)] += np.random.uniform(55, 95)
        
        # Infiltrate patch 2
        patch2_x = patch1_x + np.random.randint(-15, 16)
        patch2_y = patch1_y + np.random.randint(-20, 21)
        r2 = np.random.randint(15, 25)
        inf2_mask = ((x - patch2_x)**2 + (y - patch2_y)**2) <= r2**2
        img_array[inf2_mask & (left_lung_mask | right_lung_mask)] += np.random.uniform(40, 75)

    # Clip pixel values to valid uint8 range [0, 255]
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Smooth with slight Gaussian blur to make anatomical structures look organic
    pil_img = Image.fromarray(img_array)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.2))
    
    return pil_img

def generate_sample_dataset(samples_dir="static/samples", count_per_class=15):
    """
    Generate sample images in static/samples directory for instant testing.
    """
    os.makedirs(os.path.join(samples_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(samples_dir, "pneumonia"), exist_ok=True)
    
    created_files = {"normal": [], "pneumonia": []}
    
    # Generate Normal cases
    for i in range(count_per_class):
        img = create_synthetic_chest_xray("normal", seed=100 + i)
        path = os.path.join(samples_dir, "normal", f"normal_case_{i+1:02d}.jpg")
        img.save(path, quality=90)
        created_files["normal"].append(path)
        
    # Generate Pneumonia cases
    for i in range(count_per_class):
        img = create_synthetic_chest_xray("pneumonia", seed=200 + i)
        path = os.path.join(samples_dir, "pneumonia", f"pneumonia_case_{i+1:02d}.jpg")
        img.save(path, quality=90)
        created_files["pneumonia"].append(path)
        
    print(f"[Dataset Generator] Successfully generated {count_per_class*2} sample X-Ray images in '{samples_dir}'.")
    return created_files

if __name__ == "__main__":
    generate_sample_dataset()

