from pathlib import Path

import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

from recycle_classifier.config import DEFAULT_MODEL_PATH
from recycle_classifier.model import load_model_for_inference
from recycle_classifier.transforms import get_eval_transforms


def get_device() -> torch.device:
    """Choose the fastest available PyTorch device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_classifier(model_path: str) -> tuple[torch.nn.Module, list[str], torch.device]:
    """Load the model once and reuse it across Streamlit reruns."""
    device = get_device()
    model, class_names = load_model_for_inference(model_path, device)
    return model, class_names, device


def predict(image: Image.Image, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Run one uploaded image through the model and return class probabilities."""
    transform = get_eval_transforms()
    # Convert to RGB so grayscale or transparent uploads still match the
    # three-channel input expected by the model.
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        # Softmax converts raw logits into probabilities for each class.
        return F.softmax(logits, dim=1).squeeze(0).cpu()


# Configure the Streamlit page and collect user inputs.
st.set_page_config(page_title="Trash/Recycle Classifier")
st.title("Trash/Recycle Classifier")

model_path = st.sidebar.text_input("Model path", str(DEFAULT_MODEL_PATH))
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if not Path(model_path).exists():
    # Stop early with a training hint when there is no checkpoint to load yet.
    st.info("Train the model first, then load the saved checkpoint here.")
    st.code(f"python train.py --epochs 8 --model-path {model_path}")
    st.stop()

model, class_names, device = load_classifier(model_path)

if uploaded_file is not None:
    # Display the uploaded image first so the prediction is easy to verify.
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    probabilities = predict(image, model, device)
    top_index = int(probabilities.argmax().item())

    # Show the winning class, its confidence, and the full probability
    # distribution so close calls are visible.
    st.subheader(f"Prediction: {class_names[top_index]}")
    st.metric("Confidence", f"{probabilities[top_index].item() * 100:.1f}%")

    scores = {
        class_name: float(probabilities[index].item())
        for index, class_name in enumerate(class_names)
    }
    st.bar_chart(scores)
else:
    st.write("Upload a photo of cardboard, glass, metal, paper, plastic, or trash.")
