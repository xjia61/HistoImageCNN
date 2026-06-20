from pathlib import Path
import pandas as pd

DATA_DIR = Path("./Imgdata")

records = []

for patient_dir in DATA_DIR.iterdir():
    if not patient_dir.is_dir():
        continue

    patient_id = patient_dir.name

    for label_dir in ["0", "1"]:
        class_dir = patient_dir / label_dir
        if not class_dir.exists():
            continue

        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                records.append({
                    "path": str(img_path),
                    "patient_id": patient_id,
                    "label": int(label_dir)
                })

df = pd.DataFrame(records)

print(df.head())
print(df["label"].value_counts())
print(df["patient_id"].nunique())


from sklearn.model_selection import GroupShuffleSplit

# first split: train vs temp
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

train_idx, temp_idx = next(
    gss.split(df, y=df["label"], groups=df["patient_id"])
)

train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df = df.iloc[temp_idx].reset_index(drop=True)

# second split: val vs test
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

val_idx, test_idx = next(
    gss2.split(temp_df, y=temp_df["label"], groups=temp_df["patient_id"])
)

val_df = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

print(len(train_df), "train")
print(len(val_df), "val")
print(len(test_df), "test")


train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])
test_patients = set(test_df["patient_id"])

print("Train-Val overlap:", train_patients & val_patients)
print("Train-Test overlap:", train_patients & test_patients)
print("Val-Test overlap:", val_patients & test_patients)