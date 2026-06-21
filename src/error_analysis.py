import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_cancer_probabilities(model, data_loader, device):
    """
    Return cancer probabilities in the same order as test_loader.

    Assumption:
        class 0 = no_cancer
        class 1 = cancer
    """

    all_probs = []

    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            cancer_probs = probs[:, 1]
            all_probs.extend(cancer_probs.cpu().numpy())

    return np.array(all_probs)


def save_ensemble_error_analysis(
    resnet_model,
    efficientnet_model,
    test_loader,
    test_df,
    device,
    resnet_checkpoint,
    efficientnet_checkpoint,
    output_dir,
    resnet_weight=0.5,
    efficientnet_weight=0.5,
    threshold=0.5,
    copy_images=True,
    max_copy_per_group=200
):
    """
    Save false positive and false negative image lists for ensemble model.

    False negative:
        true label = cancer
        predicted = no_cancer

    False positive:
        true label = no_cancer
        predicted = cancer
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    false_negative_dir = output_dir / "false_negative"
    false_positive_dir = output_dir / "false_positive"

    false_negative_dir.mkdir(parents=True, exist_ok=True)
    false_positive_dir.mkdir(parents=True, exist_ok=True)

    # Reset index to make sure row order matches DataLoader order
    test_df = test_df.reset_index(drop=True).copy()

    # Load trained models
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

    # Get probabilities
    probs_resnet = get_cancer_probabilities(
        resnet_model,
        test_loader,
        device
    )

    probs_eff = get_cancer_probabilities(
        efficientnet_model,
        test_loader,
        device
    )

    assert len(test_df) == len(probs_resnet)
    assert len(test_df) == len(probs_eff)

    # Ensemble probability
    ensemble_probs = (
        resnet_weight * probs_resnet +
        efficientnet_weight * probs_eff
    )

    preds = (ensemble_probs >= threshold).astype(int)

    test_df["prob_cancer_resnet18"] = probs_resnet
    test_df["prob_cancer_efficientnet_b0"] = probs_eff
    test_df["prob_cancer_ensemble"] = ensemble_probs
    test_df["pred_label"] = preds

    test_df["true_class"] = test_df["label"].map({
        0: "no_cancer",
        1: "cancer"
    })

    test_df["pred_class"] = test_df["pred_label"].map({
        0: "no_cancer",
        1: "cancer"
    })

    # Define error types
    false_negative_df = test_df[
        (test_df["label"] == 1) &
        (test_df["pred_label"] == 0)
    ].copy()

    false_positive_df = test_df[
        (test_df["label"] == 0) &
        (test_df["pred_label"] == 1)
    ].copy()

    # Sort by confidence
    # False negatives: cancer missed with low predicted cancer probability
    false_negative_df = false_negative_df.sort_values(
        by="prob_cancer_ensemble",
        ascending=True
    )

    # False positives: no cancer called cancer with high predicted cancer probability
    false_positive_df = false_positive_df.sort_values(
        by="prob_cancer_ensemble",
        ascending=False
    )

    # Save CSV files
    fn_csv = output_dir / f"false_negative_threshold_{threshold}.csv"
    fp_csv = output_dir / f"false_positive_threshold_{threshold}.csv"
    all_csv = output_dir / f"all_predictions_threshold_{threshold}.csv"

    false_negative_df.to_csv(fn_csv, index=False)
    false_positive_df.to_csv(fp_csv, index=False)
    test_df.to_csv(all_csv, index=False)

    print(f"Saved all predictions to: {all_csv}")
    print(f"Saved false negatives to: {fn_csv}")
    print(f"Saved false positives to: {fp_csv}")

    print("\nError summary")
    print("-" * 30)
    print(f"Threshold: {threshold}")
    print(f"False negatives: {len(false_negative_df)}")
    print(f"False positives: {len(false_positive_df)}")

    # Optionally copy images for manual review
    if copy_images:
        copy_error_images(
            false_negative_df,
            false_negative_dir,
            prefix="FN",
            max_copy=max_copy_per_group
        )

        copy_error_images(
            false_positive_df,
            false_positive_dir,
            prefix="FP",
            max_copy=max_copy_per_group
        )

    return test_df, false_negative_df, false_positive_df


def copy_error_images(error_df, output_dir, prefix="ERROR", max_copy=200):
    """
    Copy selected error images into a review folder.

    File name format:
        FN_0001_prob_0.123_patient_xxx_originalname.png
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = error_df.head(max_copy)

    for i, row in subset.iterrows():
        src_path = Path(row["path"])

        if not src_path.exists():
            print(f"Missing file: {src_path}")
            continue

        prob = row["prob_cancer_ensemble"]
        patient_id = row["patient_id"]

        new_name = (
            f"{prefix}_{i:05d}_"
            f"prob_{prob:.3f}_"
            f"patient_{patient_id}_"
            f"{src_path.name}"
        )

        dst_path = output_dir / new_name

        shutil.copy2(src_path, dst_path)

    print(f"Copied {len(subset)} images to: {output_dir}")