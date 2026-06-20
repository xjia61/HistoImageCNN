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
    learning_rate=1e-6,
    weight_decay=1e-4,
    checkpoint_dir=PROJECT_ROOT / "outputs" / "checkpoints",
    model_name="resnet18"
)