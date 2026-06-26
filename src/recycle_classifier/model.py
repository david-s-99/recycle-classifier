import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, freeze_features: bool = True) -> nn.Module:
    """Create a MobileNetV3 Small classifier for the requested class count."""
    # Start from ImageNet-pretrained weights so the model already has useful
    # general visual features before learning the recycle/trash classes.
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_features:
        # Keep the convolutional feature extractor fixed during transfer
        # learning. Training then updates only the final classification head.
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    # Replace the ImageNet output layer with one sized for this dataset.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_model_for_inference(checkpoint_path: str | bytes, device: torch.device) -> tuple[nn.Module, list[str]]:
    """Load a saved checkpoint and return a ready-to-use inference model."""
    # map_location lets a checkpoint saved on one device, such as CUDA, load on
    # another available device, such as CPU or Apple MPS.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    # Recreate the architecture before loading the trained parameter values.
    model = build_model(num_classes=len(class_names), freeze_features=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    # Disable training-specific behavior such as dropout for stable predictions.
    model.eval()

    return model, class_names
