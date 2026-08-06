# ♻️ Eco-Sort
### Intelligent Waste Classification & Automated Sorting System

**A deep learning powered waste classification system using MobileNetV2 + Grad-CAM explainability**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://ecosort-app-eswthvznekunucylx9haq7.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-sukanta022/ecosort--app-181717?style=for-the-badge&logo=github)](https://github.com/sukanta022/ecosort-app)

</div>


<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white"/>
<img src="https://img.shields.io/badge/MobileNetV2-Transfer_Learning-4CAF50?style=for-the-badge"/>

<br/><br/>

---

## 📌 Overview

**Eco-Sort** is an end-to-end intelligent waste management system that:

- **Classifies** waste images into 6 categories using MobileNetV2 Transfer Learning
- **Explains** predictions visually using Grad-CAM activation heatmaps
- **Routes** each classified item to the correct sorting lane, bin, and processing facility
- **Deploys** as a real-time interactive web application on Streamlit Cloud

> **Test Accuracy: 86.77% · F1-Score: 0.85 · 6 Waste Classes · 5 Sorting Lanes**

---

## 🎯 Problem Statement

Manual waste sorting is labor-intensive, error-prone, and inconsistent. With over **2 billion tonnes** of solid waste generated globally each year, automated intelligent sorting systems are critical for improving recycling efficiency and reducing environmental impact.

Eco-Sort bridges the gap between AI research and real-world waste management by combining **accurate classification**, **visual explainability**, and **automated sorting decisions** in a single deployable system.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Deep Learning Classification** | MobileNetV2 with Transfer Learning — 86.77% test accuracy |
| **Two-Phase Training** | Feature Extraction → Fine-Tuning strategy |
| **Grad-CAM Explainability** | Visual heatmaps showing which regions drove the prediction |
| **Automated Sorting Framework** | Rule-based controller assigns bin, lane, action & destination |
| **ONNX Runtime Deployment** | Python-version-agnostic inference on Streamlit Cloud |
| **Interactive Web Application** | Clean dashboard UI with real-time classification |

---

## 🏗️ System Architecture

```
Waste Image (Input)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  Resize 224×224 · Normalize [0,1]
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   MobileNetV2       │  ImageNet pretrained · Feature Extraction
│   + Custom Head     │  Dense(256) → Dense(128) → Dense(6)
└─────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────────────────┐
│Class │  │  Grad-CAM        │  Conv_1 activation heatmap
│Label │  │  Heatmap         │
└──────┘  └──────────────────┘
   │
   ▼
┌─────────────────────┐
│  Sorting Controller │  Rule-based lookup table
│  (Decision Engine)  │  Bin · Lane · Action · Destination
└─────────────────────┘
        │
        ▼
  Web Application (Streamlit Cloud · ONNX Runtime)
```

---

## 📊 Model Performance

### Overall Metrics

| Metric | Value |
|---|---|
| Test Accuracy | **86.77%** |
| Precision (Weighted) | 0.85 |
| Recall (Weighted) | 0.85 |
| F1-Score (Weighted) | **0.85** |
| F1-Score (Macro) | 0.83 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Cardboard | 0.94 | 0.91 | **0.93** | 70 |
| Paper | 0.92 | 0.91 | **0.91** | 108 |
| Glass | 0.85 | 0.82 | 0.83 | 82 |
| Metal | 0.80 | 0.84 | 0.82 | 68 |
| Plastic | 0.75 | 0.86 | 0.81 | 74 |
| Trash | 0.85 | 0.59 | 0.69 | 29 |

> **Note:** Trash class lower performance is due to class imbalance (n=29 vs avg n=85 for other classes)

---

## ♻️ Automated Sorting Framework

Each classified item is automatically routed through the sorting pipeline:

| Waste Class | Assigned Bin | Conveyor Lane | Processing Action | Destination |
|---|---|---|---|---|
| Cardboard | Bin A | Lane 1 | Flatten & Compress | Paper Recycling Facility |
| Glass | Bin B | Lane 2 | Colour Sort & Crush | Glass Processing Plant |
| Metal | Bin C | Lane 3 | Magnetic Separation | Metal Scrap Facility |
| Paper | Bin D | Lane 1 | Stack & Bundle | Paper Recycling Facility |
| Plastic | Bin E | Lane 4 | Shred & Pelletize | Plastic Recycling Plant |
| Trash | Bin F | Lane 5 | Compact & Seal | Controlled Landfill |

---

## 🧠 Technical Approach

### Transfer Learning — Two-Phase Training

**Phase 1 — Feature Extraction** *(Selected as Final Model)*
- MobileNetV2 base frozen (trainable = False)
- Only custom classification head trained
- Optimizer: Adam (lr = 1e-3)
- Epochs: 15 with EarlyStopping (patience=5)
- **Result: 86.77% test accuracy** ✅

**Phase 2 — Fine-Tuning**
- Last 30 layers of base model unfrozen
- Optimizer: Adam (lr = 1e-5)
- Epochs: 10 with EarlyStopping
- Result: 84.22% — slight overfitting due to small dataset size ❌

> Phase 1 selected as final model based on empirical comparison.

### Data Augmentation

```python
ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.1
)
```

### Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) computes gradients of the predicted class score with respect to the final convolutional layer (`Conv_1`), producing a heatmap that highlights the spatial regions most relevant to the model's decision — transforming the system from a **black-box** into a **partially interpretable model**.

---

## 📁 Project Structure

```
ecosort-app/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
└── model/
    ├── ecosort_model.onnx      # Classification model (ONNX)
    └── ecosort_gradcam.onnx    # Grad-CAM model (ONNX)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.11+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/sukanta022/ecosort-app.git
cd ecosort-app

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Requirements

```
streamlit
onnxruntime
numpy
opencv-python-headless
matplotlib
Pillow
```

---

## 🗃️ Dataset

**Kaggle Garbage Classification Dataset**

| Split | Samples |
|---|---|
| Train | ~1,593 |
| Validation | ~342 |
| Test | 431 |
| **Total** | **~2,366** |

**Classes:** Cardboard · Glass · Metal · Paper · Plastic · Trash

---

## 💡 How to Use

1. **Open** the [Live Demo](https://ecosort-app-eswthvznekunucylx9haq7.streamlit.app)
2. **Upload** any waste image (JPG, JPEG, PNG)
3. **View** the classification result and confidence score
4. **Check** the automated sorting decision — bin, conveyor lane, and destination
5. **Explore** the Grad-CAM heatmap to understand the model's reasoning

---

## 🔬 Explainability — Grad-CAM

Unlike standard classifiers that act as black boxes, Eco-Sort provides **visual reasoning** for every prediction:

- **Original Input** — the uploaded waste image
- **Activation Heatmap** — regions the model focused on (warmer = more attention)
- **Overlay** — heatmap superimposed on the original image

This makes the system suitable for educational demonstrations, research presentations, and real-world trust-building in AI-assisted waste management.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | TensorFlow 2.15 / Keras |
| Base Architecture | MobileNetV2 (ImageNet pretrained) |
| Explainability | Grad-CAM (GradientTape) |
| Model Export | ONNX (tf2onnx) |
| Inference Runtime | ONNX Runtime |
| Web Framework | Streamlit |
| Deployment | Streamlit Cloud |
| Image Processing | OpenCV, Pillow |
| Development Environment | Google Colab |

---

## 📈 Future Work

- Address class imbalance (Trash class) via oversampling or class-weighted loss
- Extend classification taxonomy with additional waste subcategories
- Interface sorting framework with physical hardware (conveyor belts, robotic arms)
- Ensemble learning with ResNet50 and EfficientNetB0 for improved robustness
- Real-time video stream classification

---
*Built with TensorFlow · Deployed on Streamlit Cloud*

</div>
