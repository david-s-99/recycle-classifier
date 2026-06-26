from pathlib import Path


# Folder names in the dataset must match this order because ImageFolder maps
# class names to numeric labels alphabetically.
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Input size expected by the MobileNetV3 preprocessing pipeline.
IMAGE_SIZE = 224

# Default locations used by the training script and Streamlit app.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL_PATH = Path("models/mobilenetv3_small_recycle.pth")
