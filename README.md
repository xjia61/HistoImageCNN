# Histology Image Classification with CNN Transfer Learning

A local PyTorch-based deep learning project for binary histology tile classification. The goal is to classify histology image tiles into **no cancer** and **cancer** categories using CNN-based models, transfer learning, patient-level data splitting, and clinically relevant evaluation metrics.

This project focuses not only on model training, but also on understanding model behavior, preventing data leakage, comparing different CNN architectures, evaluating class-specific performance, and exploring ensemble strategies to improve cancer detection.

---

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

Label mapping:

```text
0 = no_cancer
1 = cancer
```

Each image is parsed into a dataframe with the following columns:

| Column       | Description               |
| ------------ | ------------------------- |
| `path`       | Local image file path     |
| `patient_id` | Patient identifier        |
| `label`      | 0 = no cancer, 1 = cancer |

The task is a **tile-level binary classification problem**.

---

## Why Patient-Level Split Matters

A major concern in histology image classification is **data leakage**.

Images from the same patient may share staining patterns, scanner features, tissue processing artifacts, and background characteristics. If images from the same patient are split across training and test sets, the model may learn patient-specific features instead of true cancer morphology.

To reduce this risk, this project uses **patient-level train/validation/test splitting**, so all images from the same patient appear in only one split.

```text
patient_001 → train only
patient_002 → validation only
patient_003 → test only
```

---

## Project Structure

```text
HISTOIMAG/
├── Imgdata/                  # Local image data, not committed to GitHub
├── data/
│   └── splits/               # Optional saved train/val/test CSV files
├── notebooks/
│   ├── test.py
│   ├── evaluate_test.py
│   └── plot_history.py
├── outputs/
│   ├── checkpoints/          # Saved model weights, not committed to GitHub
│   └── figures/              # Confusion matrix, training curves, ROC curves
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── split.py
│   ├── dataloader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ensemble.py
│   └── visualize.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Local Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
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

This project was run locally using **PyTorch MPS** on Apple Silicon.

---

## Data Processing

The image dataframe is built from the folder structure:

```python
from src.dataset import build_image_dataframe

df = build_image_dataframe("Imgdata")

print(df.head())
print(df["label"].value_counts())
print(df["patient_id"].nunique())
```

Example dataframe:

| path                             | patient_id  | label |
| -------------------------------- | ----------- | ----- |
| Imgdata/patient_001/0/image1.png | patient_001 | 0     |
| Imgdata/patient_001/1/image2.png | patient_001 | 1     |

---

## Patient-Level Train/Validation/Test Split

```python
from src.split import split_by_patient, check_patient_overlap

train_df, val_df, test_df = split_by_patient(df)

check_patient_overlap(train_df, val_df, test_df)
```

Expected output:

```text
Train-Val overlap: set()
Train-Test overlap: set()
Val-Test overlap: set()
Patient-level split check passed.
```

---

## DataLoader and Image Preprocessing

Images are loaded using a custom PyTorch `Dataset` and `DataLoader`.

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

Example:

```python
from src.dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    train_df,
    val_df,
    test_df,
    image_size=224,
    batch_size=32,
    num_workers=0
)
```

---

## Models

This project compares several CNN-based models:

| Model                               | Description                                             |
| ----------------------------------- | ------------------------------------------------------- |
| SimpleCNN                           | Baseline CNN trained from scratch                       |
| ResNet18                            | ImageNet-pretrained transfer learning model             |
| ResNet18 frozen backbone            | Only the final classification layer is trained          |
| ResNet18 fine-tuning                | Full or partial model fine-tuning                       |
| EfficientNet-B0                     | Lightweight ImageNet-pretrained transfer learning model |
| ResNet18 + EfficientNet-B0 Ensemble | Probability-based ensemble combining both models        |

The SimpleCNN model serves as a learning baseline. ResNet18 and EfficientNet-B0 are used as professional transfer learning baselines. The ensemble model explores whether complementary model behavior can improve cancer detection.

---

## Training

Example ResNet18 fine-tuning:

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

The best model checkpoint is saved based on the **lowest validation loss**.

This is important because validation accuracy and validation loss may not always improve together. Accuracy only measures correct versus incorrect predictions, while loss also reflects prediction confidence.

---

## Evaluation Metrics

Models are evaluated using:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion matrix
Class-specific performance
False-negative review
False-positive review
```

For this pathology-style classification task, **cancer recall** is especially important because false-negative cancer predictions are clinically concerning.

---

## Preliminary Model Results

### SimpleCNN Baseline

| Metric              |  Value |
| ------------------- | -----: |
| Train loss          | 0.3125 |
| Train accuracy      | 0.8688 |
| Validation loss     | 0.3166 |
| Validation accuracy | 0.8582 |

The SimpleCNN baseline confirmed that the local training pipeline was functional, but its performance was lower than pretrained transfer learning models.

---

### ResNet18 Transfer Learning

| Metric                   |  Value |
| ------------------------ | -----: |
| Best validation loss     | 0.2292 |
| Best validation accuracy | 0.9053 |

ResNet18 improved validation performance compared with the SimpleCNN baseline, suggesting that transfer learning is useful for histology tile classification.

---

### Lower Learning Rate ResNet18 Run

| Metric                   |  Value |
| ------------------------ | -----: |
| Best validation loss     | 0.2380 |
| Best validation accuracy | 0.9004 |

This run showed slower but stable improvement. Validation performance plateaued around 0.90 accuracy, suggesting that the model may be approaching the performance limit of the current dataset and training setup.

---

## Model Behavior

ResNet18 and EfficientNet-B0 showed different strengths:

| Model           | Observed Strength                                   |
| --------------- | --------------------------------------------------- |
| ResNet18        | Better no-cancer class performance                  |
| EfficientNet-B0 | Better cancer recall                                |
| Ensemble        | Balances cancer sensitivity and false-positive rate |

This suggests that the two pretrained models may learn complementary features from the histology tiles.

---

## Ensemble Strategy

The ensemble combines predicted cancer probabilities from ResNet18 and EfficientNet-B0.

Instead of combining hard class labels, the ensemble averages probabilities:

```text
Final cancer probability =
0.4 × ResNet18 cancer probability +
06 × EfficientNet-B0 cancer probability
```

The final class is determined by a probability threshold:

```text
if final cancer probability >= threshold:
    predict cancer
else:
    predict no_cancer
```

A lower threshold increases cancer sensitivity, while a higher threshold reduces false positives.

---

## Ensemble Test Results

### Threshold = 0.5

| Metric           |  Value |
| ---------------- | -----: |
| Accuracy         | 0.8809 |
| Cancer precision | 0.8095 |
| Cancer recall    | 0.8008 |
| Cancer F1-score  | 0.8051 |
| ROC-AUC          | 0.9442 |

Confusion matrix:

```text
[[26409  2407]
 [ 2545 10230]]
```

Interpretation:

```text
False positives: 2407
False negatives: 2545
```

Threshold 0.5 produced a more balanced model with higher overall accuracy and higher cancer precision.

---

### Threshold = 0.4

| Metric           |  Value |
| ---------------- | -----: |
| Accuracy         | 0.8751 |
| Cancer precision | 0.7681 |
| Cancer recall    | 0.8503 |
| Cancer F1-score  | 0.8071 |
| ROC-AUC          | 0.9442 |

Confusion matrix:

```text
[[25536  3280]
 [ 1913 10862]]
```

Interpretation:

```text
False positives: 3280
False negatives: 1913
```

Lowering the threshold from 0.5 to 0.4 increased cancer recall from **0.8008 to 0.8503** and reduced false-negative cancer predictions from **2545 to 1913**, at the cost of more false positives.

---

## Threshold Comparison

| Threshold | Accuracy | Cancer Precision | Cancer Recall | Cancer F1 | False Positives | False Negatives |
| --------: | -------: | ---------------: | ------------: | --------: | --------------: | --------------: |
|       0.4 |   0.8751 |           0.7681 |        0.8503 |    0.8071 |            3280 |            1913 |
|       0.5 |   0.8809 |           0.8095 |        0.8008 |    0.8051 |            2407 |            2545 |

Threshold 0.5 is more conservative and has fewer false positives. Threshold 0.4 is more cancer-sensitive and has fewer false negatives.

For a cancer detection or screening-oriented task, threshold 0.4 may be preferable because it improves cancer recall. For a balanced classifier, threshold 0.5 may be preferable.

---

## Current Interpretation

The pretrained transfer learning models outperformed the SimpleCNN baseline. ResNet18 achieved strong validation performance, while EfficientNet-B0 appeared to provide better cancer recall. The ensemble model achieved a high ROC-AUC of **0.9442**, indicating strong ability to separate cancer from no-cancer tiles.

The threshold analysis demonstrates an important clinical trade-off:

```text
Lower threshold → higher cancer recall, fewer false negatives, more false positives
Higher threshold → higher precision, fewer false positives, more false negatives
```

This project therefore emphasizes not only overall accuracy, but also cancer recall, false-negative reduction, and class-specific model behavior.

---

## Key Learning Points

This project demonstrates:

```text
Patient-level splitting to avoid data leakage
PyTorch Dataset and DataLoader design
Local CNN training using Apple MPS
Baseline CNN training from scratch
Transfer learning with ResNet18 and EfficientNet-B0
Fine-tuning strategy comparison
Validation loss versus validation accuracy interpretation
Model checkpointing
Class-specific evaluation
Cancer recall analysis
Probability-based model ensembling
Threshold tuning
False-negative and false-positive trade-off analysis
```

---

## Next Steps

Planned improvements:

1. Evaluate individual ResNet18 and EfficientNet-B0 models on the same held-out test set.
2. Add a complete model comparison table for SimpleCNN, ResNet18, EfficientNet-B0, and ensemble models.
3. Review false-negative cancer images to identify common failure patterns.
4. Save representative false-positive and false-negative examples.
5. Add ROC curve and precision-recall curve visualizations.
6. Add Grad-CAM heatmaps for model interpretability.
7. Explore partial ResNet18 fine-tuning of later layers only.
8. Test mild stain/color augmentation.
9. Consider stain normalization for histology-specific preprocessing.
10. Explore slide-level aggregation or multiple-instance learning in future work.

---

## Notes

Raw histology images, large model checkpoints, and local output files are not committed to GitHub. The repository is intended to include source code, documentation, selected figures, and summarized experiment results.
