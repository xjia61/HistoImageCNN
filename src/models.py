import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    EfficientNet_B0_Weights,

)


class SimpleCNN(nn.Module):
    """
    Simple baseline CNN for binary histology image classification.
    Input:  RGB image [batch, 3, 224, 224]
    Output: logits [batch, 2]
    """

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def get_resnet18(num_classes=2, pretrained=True, freeze_backbone=False):
    """ResNet18 transfer learning model"""
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None
    
    model = models.resnet18(weights= weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model



def get_efficientnet_b0(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    EfficientNet-B0 transfer learning model.
    """

    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def get_model(model_name="simple_cnn", num_classes=2, pretrained = True, freeze_backbone = False):
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return get_resnet18(
            num_classes= num_classes,
            pretrained=pretrained, 
            freeze_backbone=freeze_backbone
            )
    elif model_name =="efficientnet_b0":
        return get_efficientnet_b0(
            num_classes=num_classes,
            pretrained= pretrained,
            freeze_backbone= freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")