import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.visualize import plot_training_history


def main():
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" /"RESNET18_best.pt"

    history = plot_training_history(
        checkpoint_path=checkpoint_path,
        output_dir=PROJECT_ROOT / "outputs" / "figures"/"RESNET18_2",
        show=True
    )

    print(history)


if __name__ == "__main__":
    main()