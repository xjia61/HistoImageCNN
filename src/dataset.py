from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def build_image_dataframe(data_dir):
    """
    Expected structure:
    Imgdata/
        patient_001/
            0/
                image1.png
            1/
                image2.png

    label:
        0 = no cancer
        1 = cancer
    """

    data_dir = Path(data_dir)
    records = []

    for patient_dir in data_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        for label_name in ["0", "1"]:
            label_dir = patient_dir / label_name

            if not label_dir.exists():
                continue

            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    records.append({
                        "path": str(img_path),
                        "patient_id": patient_id,
                        "label": int(label_name)
                    })

    df = pd.DataFrame(records)
    return df


class HistologyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label