import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from recycle_classifier.config import CLASS_NAMES, DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, IMAGE_SIZE
from recycle_classifier.model import build_model
from recycle_classifier.transforms import get_eval_transforms, get_train_transforms


def get_device() -> torch.device:
    """Choose the fastest available PyTorch device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one full pass over the training DataLoader."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # Standard PyTorch training step: clear gradients, run the model,
        # compute loss, backpropagate, and update trainable parameters.
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        # Track weighted loss and accuracy so the epoch summary accounts for
        # the final partial batch correctly.
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate the model and collect labels for a classification report."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size
        # Store CPU lists for scikit-learn metrics after the loop.
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return total_loss / total, correct / total, y_true, y_pred


def parse_args() -> argparse.Namespace:
    """Read command-line options for dataset paths and training settings."""
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Small for trash/recycle classification.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Train MobileNetV3 on the recycle dataset and save the best checkpoint."""
    args = parse_args()
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"

    # ImageFolder expects one subdirectory per class under train/ and val/.
    train_dataset = ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = ImageFolder(val_dir, transform=get_eval_transforms())

    # Fail fast if folder names do not line up with the model's expected labels.
    if train_dataset.classes != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, got {train_dataset.classes}")
    if val_dataset.classes != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, got {val_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = get_device()
    model = build_model(num_classes=len(CLASS_NAMES), freeze_features=True).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only the classifier head is optimized because the feature extractor is frozen.
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            # Keep the best validation checkpoint so later epochs cannot replace
            # it with a worse model.
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": "mobilenet_v3_small",
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                },
                args.model_path,
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model to: {args.model_path}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    # Allows the file to be imported without immediately starting training.
    main()
