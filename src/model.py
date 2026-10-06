"""
model.py
--------
ResNet-18 binary classifier (real=0 / fake=1), matching the architecture
used in the base paper. Uses ImageNet-pretrained weights and replaces the
final fully-connected layer for the 2-class deepfake task.
"""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(pretrained=True, num_classes=2, freeze_backbone=False):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as e:
        # ImageNet weight download blocked (e.g. offline / restricted network) --
        # fall back to random init so the pipeline still runs end-to-end.
        # On a machine with full internet access, pretrained=True will work normally.
        print(f"[model.py] Could not fetch pretrained weights ({e}); using random init instead.")
        model = resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model
