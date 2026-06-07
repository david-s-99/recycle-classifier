# Trash/Recycle Classifier

Image classifier for six waste classes using pretrained MobileNetV3 Small.

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
PYTHONPATH=src python train.py --epochs 8
```

The training script:

- loads pretrained MobileNetV3 Small
- freezes the feature extractor
- replaces the final classifier layer with a 6-class layer
- trains only the classifier head
- saves the best checkpoint to `models/mobilenetv3_small_recycle.pth`

## Streamlit Demo

```bash
PYTHONPATH=src streamlit run app.py
```

