# Histology Image CNN Classification

A local PyTorch-based deep learning project for binary histology image classification. The goal is to classify histology image tiles into **no cancer** and **cancer** categories using convolutional neural networks and transfer learning.

This project focuses on building a reproducible image classification pipeline, understanding key model parameters, comparing CNN architectures, and evaluating model performance using clinically relevant metrics.

## Project Overview

The dataset is organized by patient ID and class label:

```text
Imgdata/
├── patient_001/
│   ├── 0/   # no cancer
│   └── 1/   # cancer
├── patient_002/
│   ├── 0/
│   └── 1/
```

Each image is converted into a dataframe containing:

```text
image_path | patient_id | label
```

Label mapping:

```text
0 = no_cancer
1 = cancer
```

To avoid data leakage, the dataset is split at the **patient level**, meaning images from the same patient are never shared across training, validation, and test sets.

## Why Patient-Level Split Matters

In histology image classification, images from the same patient may share staining patterns, scanner artifacts, tissue processing features, and background characteristics. If images from the same patient appear in both training and test sets, the model may learn patient-specific artifacts instead of true cancer morphology.

Therefore, this project uses patient-level train/validation/test splitting.

## Project Structure

```text
HISTOIMAG/
├── Imgdata/                  # local image data, not committed to GitHub
├── data/
│   └── splits/               # saved train/val/test CSV files
├── notebooks/
│   ├── test.py
│   ├── evaluate_test.py
│   └── plot_history.py
├── outputs/
│   ├── checkpoints/          # saved model weights, not committed
│   └── figures/              # training curves, confusion matrix
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── split.py
│   ├── dataloader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Local Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn pillow tqdm
pip install jupyter ipykernel
```

Check device availability:

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
```

This project was run locally using Apple GPU acceleration through PyTorch MPS.

## Data Processing

The image dataframe is generated from the folder structure:

```python
from src.dataset import build_image_dataframe

df = build_image_dataframe("Imgdata")

print(df.head())
print(df["label"].value_counts())
print(df["patient_id"].nunique())
```

The dataframe contains:

| Column       | Description               |
| ------------ | ------------------------- |
| `path`       | Path to image file        |
| `patient_id` | Patient identifier        |
| `label`      | 0 = no cancer, 1 = cancer |

## Patient-Level Data Split

```python
from src.split import split_by_patient, check_patient_overlap

train_df, val_df, test_df = split_by_patient(df)

check_patient_overlap(train_df, val_df, test_df)
```

Expected result:

```text
Train-Val overlap: set()
Train-Test overlap: set()
Val-Test overlap: set()
Patient-level split check passed.
```

## DataLoader

Images are resized, augmented, normalized, and loaded into PyTorch DataLoaders.

Training transformations include:

```text
Resize
RandomHorizontalFlip
RandomVerticalFlip
RandomRotation
ToTensor
Normalize
```

Validation and test images use deterministic preprocessing without random augmentation.

## Models

This project compares multiple CNN-based approaches:

| Model                        | Description                                 |
| ---------------------------- | ------------------------------------------- |
| SimpleCNN                    | Baseline CNN trained from scratch           |
| ResNet18                     | ImageNet-pretrained transfer learning model |
| ResNet18 frozen backbone     | Only final classifier layer is trained      |
| ResNet18 partial fine-tuning | Later layers and classifier are fine-tuned  |

## Training

Example training command in Python:

```python
from src.models import get_model
from src.train import train_model

model = get_model(
    model_name="resnet18",
    num_classes=2,
    pretrained=True,
    freeze_backbone=False
)

model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    learning_rate=1e-5,
    weight_decay=1e-4,
    checkpoint_dir="outputs/checkpoints",
    model_name="resnet18"
)
```

The best model is saved based on the lowest validation loss.

## Evaluation

The trained model is evaluated on the held-out test set using:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion matrix
```

Example evaluation:

```python
from src.models import get_model
from src.train import get_device
from src.evaluate import evaluate_model

device = get_device()

model = get_model(
    model_name="resnet18",
    num_classes=2,
    pretrained=False
)

metrics = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device,
    checkpoint_path="outputs/checkpoints/resnet18_best.pt",
    output_dir="outputs/figures",
    class_names=["no_cancer", "cancer"]
)
```

## Preliminary Results

### Simple CNN Baseline

| Metric              |  Value |
| ------------------- | -----: |
| Train accuracy      | 0.8688 |
| Validation accuracy | 0.8582 |
| Train loss          | 0.3125 |
| Validation loss     | 0.3166 |

### ResNet18 Transfer Learning

| Metric                   |  Value |
| ------------------------ | -----: |
| Best validation loss     | 0.2292 |
| Best validation accuracy | 0.9053 |

The pretrained ResNet18 model improved validation performance compared with the SimpleCNN baseline, suggesting that transfer learning is useful for histology tile classification.

### Lower Learning Rate ResNet18 Run

| Metric                   |  Value |
| ------------------------ | -----: |
| Best validation loss     | 0.2380 |
| Best validation accuracy | 0.9004 |

This run showed slower but stable improvement. Validation performance plateaued around 0.90 accuracy, suggesting that the model may be approaching the performance limit of the current dataset and training setup.

### Efficientnet_b0 Transfer Learning

| Metric                   |  Value |
| ------------------------ | -----: |
| Best validation loss     | 0.2366 |
| Best validation accuracy | 0.9003 |

The pretrained Efficientnet_b0 model had simily validation performance ocompared with the ResNet18 model, but improved the recall score which is more importiant in Histology Image Classification.


## Interpretation

The model reached approximately 0.90 validation accuracy with ResNet18 transfer learning. Validation loss and accuracy showed small fluctuations after reaching this level, which likely represents a performance plateau rather than severe overfitting.

The close training and validation performance suggests reasonable generalization. Further improvement may require:

```text
More patients
Better tile-level labels
Misclassified image review
Stain normalization
Mild color augmentation
Partial fine-tuning
Dropout regularization
Class imbalance handling
```

## Next Steps

Planned improvements:

1. Evaluate all models on the same held-out test set.
2. Compare SimpleCNN, ResNet18 frozen, and ResNet18 fine-tuned models.
3. Review false positive and false negative images.
4. Save misclassified image examples for error analysis.
5. Add ROC curve and precision-recall curve.
6. Add experiment tracking table.
7. Explore partial fine-tuning of ResNet18 layer4 and classifier head.
8. Consider stain normalization for histology-specific preprocessing.

## Key Learning Points

This project demonstrates:

```text
Patient-level split to avoid data leakage
PyTorch Dataset and DataLoader design
CNN baseline training
Transfer learning with pretrained ResNet18
Fine-tuning strategy comparison
Validation loss vs validation accuracy interpretation
Model checkpointing
Confusion matrix and ROC-AUC evaluation
Histology-specific error analysis
```

## Notes

Large image files, model checkpoints, and raw data are not committed to GitHub. Only source code, documentation, and selected result figures are included.
