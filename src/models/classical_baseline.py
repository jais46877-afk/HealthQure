"""
src/models/classical_baseline.py
==============================================================================
Pure Classical Baseline Model - Frozen ResNet-18 + MLP head
==============================================================================

Identical ResNet-18 backbone (frozen) as the Hybrid QNN, but the quantum
layer is replaced by a standard MLP classifier:

    Linear(512 -> 64) -> ReLU -> Linear(64 -> 3)

This enables a fair apple-to-apple comparison:
  - Same backbone weights & frozen layers
  - Same number of output classes
  - No quantum overhead
==============================================================================
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ==============================================================================
# Helper
# ==============================================================================

def _frozen_resnet18_backbone() -> nn.Module:
    """
    Load the default pre-trained ResNet-18, freeze all conv parameters,
    and return it with the fc head replaced by Identity (output: Bx512).
    """
    backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

    for param in backbone.parameters():
        param.requires_grad = False

    backbone.fc = nn.Identity()   # output: (B, 512)
    return backbone


# ==============================================================================
# Classical Baseline
# ==============================================================================

class ClassicalBaseline(nn.Module):
    """
    Pure classical ResNet-18 baseline for breast cancer 3-class classification.

    Architecture:
        [Frozen ResNet-18]  ->  (B, 512)
             |
        Linear(512 -> 64) -> ReLU -> Linear(64 -> 3)
             |
        class logits  (B, 3)

    Parameters
    ----------
    n_classes : int
        Number of output classes (default 3).
    """

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()

        self.n_classes = n_classes

        # Frozen backbone
        self.backbone = _frozen_resnet18_backbone()

        # Trainable classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    # --------------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 3, 224, 224)

        Returns
        -------
        logits : torch.Tensor  shape (B, n_classes)
        """
        features = self.backbone(x)          # (B, 512)
        logits   = self.classifier(features) # (B, 3)
        return logits

    # --------------------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """Return trainable and total parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def __repr__(self) -> str:
        info = self.count_parameters()
        return "ClassicalBaseline(n_classes={}, trainable_params={:,})".format(
            self.n_classes, info["trainable"]
        )


# ==============================================================================
# Quick smoke-test
# ==============================================================================

if __name__ == "__main__":
    model = ClassicalBaseline()
    print(model)

    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print("Output shape  : {}".format(out.shape))
    print("Output logits :\n{}".format(out))
