from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "archive (1)" / "EuroSAT" / "2750"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
SEED = 42

MODEL_PATH = MODEL_DIR / "crop_classifier.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
HISTORY_PATH = REPORT_DIR / "training_history.csv"
PLOT_PATH = REPORT_DIR / "training_curves.png"
