import torch
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_cancer_probabilities(model, data_loader, device):
    """
    Return true labels and predicted probability for cancer class.
    Assumes:
        class 0 = no_cancer
        class 1 = cancer
    """

    all_labels = []
    all_probs = []

    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            cancer_probs = probs[:, 1]

            all_labels.extend(labels.numpy())
            all_probs.extend(cancer_probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def evaluate_ensemble(
    resnet_model,
    efficientnet_model,
    test_loader,
    device,
    resnet_checkpoint,
    efficientnet_checkpoint,
    resnet_weight=0.5,
    efficientnet_weight=0.5,
    threshold=0.5,
    class_names=None
):
    """
    Ensemble ResNet18 and EfficientNet-B0 by averaging cancer probabilities.
    """

    if class_names is None:
        class_names = ["no_cancer", "cancer"]

    resnet_model = load_checkpoint(
        resnet_model,
        resnet_checkpoint,
        device
    )

    efficientnet_model = load_checkpoint(
        efficientnet_model,
        efficientnet_checkpoint,
        device
    )

    labels_resnet, probs_resnet = get_cancer_probabilities(
        resnet_model,
        test_loader,
        device
    )

    labels_eff, probs_eff = get_cancer_probabilities(
        efficientnet_model,
        test_loader,
        device
    )

    assert np.array_equal(labels_resnet, labels_eff)

    y_true = labels_resnet

    ensemble_probs = (
        resnet_weight * probs_resnet +
        efficientnet_weight * probs_eff
    )

    y_pred = (ensemble_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, ensemble_probs)
    cm = confusion_matrix(y_true, y_pred)

    print("\nEnsemble Test Metrics")
    print("-" * 30)
    print(f"Threshold:  {threshold}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "threshold": threshold,
        "resnet_weight": resnet_weight,
        "efficientnet_weight": efficientnet_weight
    }

    return metrics