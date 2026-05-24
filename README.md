# Classify Crops Using Satellite Images

This project trains an image classification model to identify satellite land-cover/crop-scene classes from the EuroSAT Sentinel-2 image dataset.

The default dataset path is:

```text
archive (1)/EuroSAT/2750/
```

Each folder is treated as one class label. The current Sentinel-2 RGB dataset contains 10 classes and 27,000 JPG images, including crop-related classes such as `AnnualCrop` and `PermanentCrop`.

## Project Structure

```text
.
|-- archive (1)/EuroSAT/2750/ # Sentinel-2 RGB JPG class folders
|-- app.py                   # Streamlit app for satellite image upload and prediction
|-- dataset_overview.py      # Creates a class/image-count report
|-- predict.py               # Command-line prediction script
|-- train.py                 # Model training script
|-- requirements.txt         # Python dependencies
|-- src/                     # Reusable project code
|-- models/                  # Trained model and class labels
`-- reports/                 # Training reports and plots
```

## Setup

Install 64-bit Python 3.11 or 3.12, then run:

```powershell
python check_environment.py
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows, this project installs `tensorflow-intel`, the native Windows TensorFlow build. If you created the virtual environment with Python 3.13, Python 3.14, or 32-bit Python, delete `.venv`, install 64-bit Python 3.11 or 3.12, and create the virtual environment again.

For example, after installing Python 3.12:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python check_environment.py
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Check the Dataset

```powershell
python dataset_overview.py
python find_tensorflow_bad_images.py
```

This writes:

```text
reports/dataset_overview.csv
```

## Train the Model

```powershell
python train.py --epochs 20
```

For a quick CPU test:

```powershell
python train.py --epochs 2
```

The training script uses MobileNetV2 transfer learning with image augmentation. The default image size is 96x96, which matches an official MobileNetV2 pretrained input size and usually improves accuracy compared with 64x64. It saves:

```text
models/crop_classifier.keras
models/class_names.json
reports/training_history.csv
reports/training_curves.png
```

If your machine cannot download ImageNet weights, train with random initialization:

```powershell
python train.py --epochs 20 --base-weights none
```

## Predict One Satellite Image

```powershell
python predict.py "archive (1)\EuroSAT\2750\AnnualCrop\AnnualCrop_1.jpg"
```

## Run the Web App

```powershell
streamlit run app.py
```

Then upload any Sentinel-2 style satellite image in the browser to get the top predictions.

## Notes

EuroSAT includes both crop and non-crop land-cover classes. For a crop-only classifier, train with only crop folders such as `AnnualCrop` and `PermanentCrop`, or add more labeled crop-type folders.
