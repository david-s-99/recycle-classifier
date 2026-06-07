from pathlib import Path


CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_SIZE = 224
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL_PATH = Path("models/mobilenetv3_small_recycle.pth")

