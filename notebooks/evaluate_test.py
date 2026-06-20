import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dataset import build_image_dataframe
from src.split import split_by_patient, check_patient_overlap
from src.dataloader import create_dataloaders
from src.models import get_model
from src.train import get_device
from src.evaluate import evaluate_model


def main():
    print("Project root:", PROJECT_ROOT)

    # 1. Build dataframe
    df = build_image_dataframe(PROJECT_ROOT / "Imgdata")

    # 2. Recreate the same patient-level split
    # Important: use the same random_state as training
    train_df, val_df, test_df = split_by_patient(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    check_patient_overlap(train_df, val_df, test_df)

    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)

    # 3. Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df,
        val_df,
        test_df,
        image_size=224,
        batch_size=32,
        num_workers=0
    )

    # 4. Rebuild the same model architecture
    model = get_model("efficientnet_b0", num_classes=2)

    # 5. Load checkpoint and evaluate on test set
    device = get_device()

    #checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "simple_cnn_best.pt"
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "efficientnet_b0_best.pt"


    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        output_dir=PROJECT_ROOT / "outputs" / "figures"/"efficientnet_3",
        class_names=["no_cancer", "cancer"]
    )

    print(metrics)


if __name__ == "__main__":
    main()