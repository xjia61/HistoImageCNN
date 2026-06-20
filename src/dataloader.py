from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import HistologyDataset


def get_transforms(image_size=224):
    """
    Define image preprocessing and augmentation.
    """

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def create_dataloaders(
    train_df,
    val_df,
    test_df,
    image_size=224,
    batch_size=32,
    num_workers=0
):
    """
    Create PyTorch DataLoaders from train/val/test dataframes.

    num_workers=0 is safest on Mac / MPS.
    """

    train_transform, eval_transform = get_transforms(image_size=image_size)

    train_dataset = HistologyDataset(train_df, transform=train_transform)
    val_dataset = HistologyDataset(val_df, transform=eval_transform)
    test_dataset = HistologyDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader