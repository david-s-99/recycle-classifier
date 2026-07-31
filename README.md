# Trash/Recycle Classifier

Image classifier for six waste classes using pretrained MobileNetV3 Small.

The repository includes a trained checkpoint with **86.1% validation accuracy**,
so the demo works immediately after setup without retraining.

## Repo Structure

```txt
.
├── app.py
├── train.py
├── requirements.txt
├── data/
│   ├── train/
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── val/
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
├── models/
│   └── mobilenetv3_small_recycle.pth
├── reports/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   └── misclassified_examples.png
└── src/
    └── recycle_classifier/
        ├── config.py
        ├── model.py
        └── transforms.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
PYTHONPATH=src python train.py --epochs 8 --seed 42
```

The training script:

- loads pretrained MobileNetV3 Small
- freezes the feature extractor
- replaces the final classifier layer with a 6-class layer
- trains only the classifier head
- saves the best checkpoint to `models/mobilenetv3_small_recycle.pth`
- reloads the best checkpoint for final evaluation
- writes per-class precision, recall, and F1 scores to `reports/metrics.json`
- creates a confusion matrix, class-distribution chart, and example mistakes in `reports/`

Python, NumPy, PyTorch, and DataLoader workers are seeded. Dependencies are
pinned in `requirements.txt` so repeated runs use the same software versions.

## Streamlit Demo

```bash
PYTHONPATH=src streamlit run app.py
```

The included checkpoint is loaded automatically. Upload a JPG, PNG, or WebP
image to see the predicted waste class, confidence, and all class probabilities.
