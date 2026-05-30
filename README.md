# 🛰️ AgriVision — Satellite Crop Intelligence Platform

> **AI-powered land-cover and crop classification from Sentinel-2 multispectral satellite imagery.**  
> Built with TensorFlow Transfer Learning + Streamlit.

---

## 📌 Project Overview

**AgriVision** is an end-to-end deep learning system that classifies land cover and crop types from European Space Agency (ESA) **Sentinel-2 RGB satellite images**. It uses **Transfer Learning** on the EuroSAT benchmark dataset to train a convolutional neural network that can distinguish 10 land-use categories — including agricultural crop types, forests, water bodies, and urban zones.

The project includes:
- A **training pipeline** supporting multiple CNN architectures (MobileNetV2, ResNet50, EfficientNetB0)
- A **command-line prediction tool** for quick inference
- A **full-featured Streamlit web dashboard** (AgriVision) for interactive satellite image analysis

---

## 🧠 Model & Algorithm Details

### Architecture — Transfer Learning (CNN)

The classifier is built using **Transfer Learning** on ImageNet-pretrained Convolutional Neural Networks. The base model's convolutional layers are frozen, and a custom classification head is trained on top.

| Component | Details |
|---|---|
| **Default Architecture** | MobileNetV2 (lightweight, mobile-optimised CNN) |
| **Alternative Architectures** | ResNet50, EfficientNetB0 |
| **Pretrained Weights** | ImageNet (1.2M images, 1000 classes) |
| **Input Resolution** | 96 × 96 pixels (RGB, 3 channels) |
| **Classification Head** | GlobalAveragePooling2D → Dropout(0.3) → Dense(softmax) |
| **Loss Function** | Categorical Cross-Entropy |
| **Optimiser** | Adam (default lr = 1e-3) |
| **Metrics** | Accuracy, Precision, Recall |

### MobileNetV2 (Default)

MobileNetV2 uses **depthwise separable convolutions** and **inverted residuals with linear bottlenecks**, making it ~8× faster and smaller than VGG-16 while maintaining competitive accuracy. Ideal for running on CPU-only machines.

### ResNet50 (Optional)

A 50-layer **Residual Network** using skip connections to avoid vanishing gradients. Better accuracy ceiling, but higher compute cost.

### EfficientNetB0 (Optional)

Uses **compound scaling** (depth + width + resolution) for best accuracy-per-FLOP. Recommended if GPU is available.

### Data Augmentation Pipeline

Applied at training time to improve generalisation on satellite imagery:

| Augmentation | Value |
|---|---|
| Random Horizontal Flip | Enabled |
| Random Rotation | ±8% |
| Random Zoom | ±12% |
| Random Contrast | ±10% |

### Training Strategy & Callbacks

| Callback | Purpose |
|---|---|
| `ModelCheckpoint` | Saves best model by `val_accuracy` |
| `EarlyStopping` | Stops training if `val_loss` stagnates for 5 epochs |
| `ReduceLROnPlateau` | Cuts learning rate by 0.3× if `val_loss` stagnates for 2 epochs |
| `Class Weights` | Computed from class frequencies to handle any class imbalance |

---

## 🗂️ Dataset — EuroSAT (Sentinel-2)

| Property | Details |
|---|---|
| **Source** | European Space Agency (ESA) Sentinel-2 satellite |
| **Total Images** | 27,000 labelled JPEG images |
| **Image Size** | 64×64 px (resized to 96×96 during training) |
| **Classes** | 10 land-cover categories |
| **Split** | 80% training / 20% validation |
| **Download** | [Kaggle — Sentinel-2 Satellite Imagery](https://www.kaggle.com/datasets/gallo33henrique/sentinel-2-satellite-imagery) |

### Land Cover Classes

| Class | Description |
|---|---|
| `AnnualCrop` | Seasonal farmland (wheat, maize, etc.) |
| `PermanentCrop` | Orchards, vineyards, olive groves |
| `Pasture` | Grasslands, grazing meadows |
| `Forest` | Dense tree canopy |
| `HerbaceousVegetation` | Shrubs, wild grasses |
| `Industrial` | Factories, warehouses |
| `Residential` | Urban housing areas |
| `Highway` | Roads and transport corridors |
| `River` | Inland water bodies — rivers |
| `SeaLake` | Large water bodies — sea and lakes |

---

## 🏗️ Project Structure

```
ml project/
├── app.py                        # AgriVision Streamlit dashboard (main UI)
├── train.py                      # Model training script
├── predict.py                    # Command-line inference script
├── dataset_overview.py           # Dataset statistics report generator
├── find_tensorflow_bad_images.py # Dataset validation / bad image scanner
├── check_environment.py          # Python / TF environment checker
├── requirements.txt              # Python dependencies
│
├── src/                          # Core reusable modules
│   ├── config.py                 # Paths, hyperparameters (IMAGE_SIZE=96, BATCH_SIZE=64)
│   ├── data.py                   # Dataset loading, augmentation, class weights
│   └── model.py                  # Model builder (MobileNetV2 / ResNet50 / EfficientNetB0)
│
├── models/                       # Saved model artefacts (auto-created on train)
│   ├── crop_classifier.keras     # Trained Keras model
│   ├── class_names.json          # Ordered list of class labels
│   └── model_metadata.json       # Arch name, image size, class names
│
├── reports/                      # Training reports (auto-created on train)
│   ├── training_history.csv      # Epoch-by-epoch accuracy / loss log
│   ├── training_curves.png       # Accuracy & loss curve plots
│   └── dataset_overview.csv      # Per-class image counts
│
├── archive (1)/EuroSAT/2750/     # EuroSAT dataset (10 class subfolders)
├── sample_images/                # Sample images for testing (including video bg)
├── static/                       # Streamlit static file serving (dashboard video)
└── .streamlit/config.toml        # Streamlit theme + static serving config
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python** 3.11 or 3.12 (64-bit only — TensorFlow does not support 3.13+ yet)
- **Windows**: uses `tensorflow-intel` (native optimised build)
- **Linux/macOS**: uses standard `tensorflow`

### 2. Environment Setup

```powershell
# Check Python version and environment
python check_environment.py

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

> **Python version mismatch?** If you used Python 3.13 or a 32-bit install:
> ```powershell
> deactivate
> Remove-Item -Recurse -Force .venv
> py -3.12 -m venv .venv
> .\.venv\Scripts\Activate.ps1
> python -m pip install --upgrade pip setuptools wheel
> pip install -r requirements.txt
> ```

### 3. Validate the Dataset

```powershell
python dataset_overview.py
python find_tensorflow_bad_images.py
```

Outputs `reports/dataset_overview.csv` with per-class image counts.

### 4. Train the Model

```powershell
# Full training (20 epochs, MobileNetV2, ImageNet weights)
python train.py --epochs 20

# Quick CPU smoke-test (2 epochs)
python train.py --epochs 2

# Use ResNet50 instead of MobileNetV2
python train.py --epochs 20 --arch resnet50

# Use EfficientNetB0
python train.py --epochs 20 --arch efficientnetb0

# Train without ImageNet weights (random init, slower convergence)
python train.py --epochs 20 --base-weights none

# Custom learning rate
python train.py --epochs 30 --learning-rate 0.0005
```

**Training outputs:**
```
models/crop_classifier.keras      ← Best model checkpoint
models/class_names.json           ← Class label list
models/model_metadata.json        ← Architecture metadata
reports/training_history.csv      ← Epoch logs (acc, val_acc, loss, val_loss)
reports/training_curves.png       ← Accuracy & loss plots
```

### 5. Predict a Single Image

```powershell
python predict.py "archive (1)\EuroSAT\2750\AnnualCrop\AnnualCrop_1.jpg"
```

### 6. Launch the AgriVision Dashboard

```powershell
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ AgriVision Dashboard Features

The Streamlit web app provides a full satellite intelligence platform with:

| Page | Features |
|---|---|
| **Dashboard Overview** | Animated aerial video background, live stats, feature showcase, orbital map, technology timeline |
| **Upload Satellite Image** | Upload JPG/PNG → instant prediction with confidence bar, class description, agronomic insights |
| **AI Analysis** | NDVI index chart, land-use probability bar chart, soil composition, image metadata |
| **Crop Predictions** | 6-panel diagnostic grid: predicted class, confidence gauge, soil radar, NDVI index, yield trend, weather suitability |
| **AI Chat Assistant** | Rule-based agronomy chatbot (soil pH, nitrogen, yield, satellite terminology) |
| **Reports & History** | Download PDF report, view session prediction log, multi-file batch classification |
| **System Settings** | NDVI calibration slider, live model metadata display, architecture info |
| **About Project** | Project description, tech stack, dataset reference |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `tensorflow-intel` / `tensorflow` | ≥ 2.16 | Deep learning framework (CNN training + inference) |
| `numpy` | ≥ 1.26 | Numerical computing, array ops |
| `pillow` | ≥ 10.0 | Image loading and preprocessing |
| `matplotlib` | ≥ 3.8 | Training curve plots |
| `pandas` | ≥ 2.2 | Dataset stats, history CSV, batch results |
| `scikit-learn` | ≥ 1.4 | Class weight computation, metrics |
| `streamlit` | ≥ 1.34 | Web dashboard UI framework |
| `plotly` | (via streamlit) | Interactive charts: gauge, radar, line charts |
| `fpdf2` | (via streamlit) | PDF report generation |

---

## 🔬 Technical Notes

- **Transfer Learning approach**: All base model weights are **frozen** during training. Only the custom classification head (GAP → Dropout → Dense) is trained. This makes training fast even on CPU.
- **Class weights**: Automatically computed as `total / (n_classes × class_count)` to handle any imbalance.
- **EarlyStopping patience = 5**: Prevents overfitting; training halts if validation loss does not improve for 5 consecutive epochs.
- **Best model saved**: `ModelCheckpoint` monitors `val_accuracy` and saves only the epoch with the highest validation accuracy.
- **Image normalisation**: Each architecture uses its own `preprocess_input` function (MobileNetV2 maps to `[-1, 1]`, ResNet50 uses channel mean subtraction).

---

## 📄 License

This project uses the [EuroSAT dataset](https://github.com/phelber/EuroSAT) published under the MIT license. Satellite imagery courtesy of the European Space Agency (ESA) Copernicus Programme.
