from torchvision import transforms

from recycle_classifier.config import IMAGE_SIZE


def get_train_transforms() -> transforms.Compose:
    """Build the preprocessing pipeline used while training.

    Training transforms include light random augmentation so the model sees
    varied versions of the same class and learns features that generalize
    beyond the exact training photos.
    """
    return transforms.Compose(
        [
            # MobileNet expects a fixed-size image tensor, so resize every input
            # image before applying augmentation and normalization.
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            # These are ImageNet channel statistics, matching the pretrained
            # MobileNet weights used by the classifier.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Build the deterministic preprocessing pipeline for validation/inference."""
    return transforms.Compose(
        [
            # Evaluation must be repeatable, so it only resizes, tensorizes, and
            # normalizes instead of applying random augmentation.
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
