import torch
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(
    checkpoint_path,
    output_dir="outputs/figures",
    show=True
):
    """
    Load training history from checkpoint and plot:
    1. train loss vs val loss
    2. train accuracy vs val accuracy
    """

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    history = checkpoint["history"]

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    epochs = range(1, len(train_loss) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    loss_path = output_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_acc, label="Train accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    acc_path = output_dir / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved loss curve to: {loss_path}")
    print(f"Saved accuracy curve to: {acc_path}")

    return history