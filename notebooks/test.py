import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.append(str(PROJECT_ROOT))

print("Current directory:", Path.cwd())
print("Project root:", PROJECT_ROOT)
"""
from src.dataset import build_image_dataframe

df = build_image_dataframe(PROJECT_ROOT / "Imgdata")

print(df.head())
print(df.shape)
print(df["label"].value_counts())
print("Number of patients:", df["patient_id"].nunique())


from src.dataset import build_image_dataframe
from src.split import split_by_patient, check_patient_overlap

df = build_image_dataframe(PROJECT_ROOT / "Imgdata")

train_df, val_df, test_df = split_by_patient(df)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

check_patient_overlap(train_df, val_df, test_df)


split_dir = PROJECT_ROOT / "data" / "splits"
split_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(split_dir / "train.csv", index=False)
val_df.to_csv(split_dir / "val.csv", index=False)
test_df.to_csv(split_dir / "test.csv", index=False)
"""

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


#test cnn simple model
import torch
from src.models import get_model

model = get_model("simple_cnn", num_classes=2)

images, labels = next(iter(train_loader))

outputs = model(images)

print("Images:", images.shape)
print("Outputs:", outputs.shape)
print("Labels:", labels.shape)

from src.models import get_model
from src.train import train_model

model = get_model("simple_cnn", num_classes=2)

model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=5,
    learning_rate=1e-4,
    weight_decay=1e-4,
    checkpoint_dir=PROJECT_ROOT / "outputs" / "checkpoints",
    model_name="simple_cnn"
)



from src.models import get_model
from src.train import get_device
from src.evaluate import evaluate_model

device = get_device()

model = get_model("simple_cnn", num_classes=2)

checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "simple_cnn_best.pt"

metrics = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device,
    checkpoint_path=checkpoint_path,
    output_dir=PROJECT_ROOT / "outputs" / "figures",
    class_names=["no_cancer", "cancer"]
)