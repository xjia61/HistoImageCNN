import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import build_image_dataframe
from src.split import split_by_patient, check_patient_overlap
from src.dataloader import create_dataloaders
from src.models import get_model
from src.train import get_device
from src.error_analysis import save_ensemble_error_analysis


def main():
    print("Project root:", PROJECT_ROOT)

    # Rebuild dataframe and split
    df = build_image_dataframe(PROJECT_ROOT / "Imgdata")

    train_df, val_df, test_df = split_by_patient(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    check_patient_overlap(train_df, val_df, test_df)

    # Recreate dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df,
        val_df,
        test_df,
        image_size=224,
        batch_size=32,
        num_workers=0
    )

    device = get_device()
    print("Using device:", device)

    # Rebuild model structures
    resnet_model = get_model(
        model_name="resnet18",
        num_classes=2,
        pretrained=False
    )

    efficientnet_model = get_model(
        model_name="efficientnet_b0",
        num_classes=2,
        pretrained=False
    )

    # Checkpoints
    resnet_checkpoint = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet18_best.pt"
    efficientnet_checkpoint = PROJECT_ROOT / "outputs" / "checkpoints" / "efficientnet_b0_best.pt"

    # Save false negatives and false positives
    test_pred_df, false_negative_df, false_positive_df = save_ensemble_error_analysis(
        resnet_model=resnet_model,
        efficientnet_model=efficientnet_model,
        test_loader=test_loader,
        test_df=test_df,
        device=device,
        resnet_checkpoint=resnet_checkpoint,
        efficientnet_checkpoint=efficientnet_checkpoint,
        output_dir=PROJECT_ROOT / "outputs" / "error_analysis" / "ensemble_threshold_0.4",
        resnet_weight=0.4,
        efficientnet_weight=0.6,
        threshold=0.5,
        copy_images=True,
        max_copy_per_group=200
    )

    print("\nFalse negative examples:")
    print(false_negative_df[[
        "path",
        "patient_id",
        "label",
        "pred_label",
        "prob_cancer_ensemble"
    ]].head(20))

    print("\nFalse positive examples:")
    print(false_positive_df[[
        "path",
        "patient_id",
        "label",
        "pred_label",
        "prob_cancer_ensemble"
    ]].head(20))


if __name__ == "__main__":
    main()