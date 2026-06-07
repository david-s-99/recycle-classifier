import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, freeze_features: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_model_for_inference(checkpoint_path: str | bytes, device: torch.device) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names), freeze_features=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names

