import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.append(str(PROJECT_ROOT))

print("Current directory:", Path.cwd())
print("Project root:", PROJECT_ROOT)


from src.dataset import build_image_dataframe
from src.split import split_by_patient, check_patient_overlap
from src.dataloader import create_dataloaders

df = build_image_dataframe(PROJECT_ROOT / "Imgdata")

train_df, val_df, test_df = split_by_patient(df)

check_patient_overlap(train_df, val_df, test_df)

train_loader, val_loader, test_loader = create_dataloaders(
    train_df,
    val_df,
    test_df,
    image_size=224,
    batch_size=32
)

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Labels:", labels)

from src.models import get_model
from src.train import get_device
from src.ensemble import evaluate_ensemble

device = get_device()

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

resnet_checkpoint = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet18_best.pt"
efficientnet_checkpoint = PROJECT_ROOT / "outputs" / "checkpoints" / "efficientnet_b0_best.pt"

metrics = evaluate_ensemble(
    resnet_model=resnet_model,
    efficientnet_model=efficientnet_model,
    test_loader=test_loader,
    device=device,
    resnet_checkpoint=resnet_checkpoint,
    efficientnet_checkpoint=efficientnet_checkpoint,
    resnet_weight=0.4,
    efficientnet_weight=0.6,
    threshold=0.5,
    class_names=["no_cancer", "cancer"]
)