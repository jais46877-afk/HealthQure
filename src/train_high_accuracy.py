"""
src/train_high_accuracy.py
==============================================================================
Balanced High-Accuracy Hybrid QNN Trainer (v3 - Focal Loss & Macro-F1 Optimization)
==============================================================================
Fixes class imbalance and model collapse by:
  1. Using Focal Loss (gamma=2.0) with exact inverse-frequency class weights
  2. Balanced Accuracy & Macro-F1 checkpoint selection (NOT raw accuracy)
  3. Unfreezing ResNet layer3 + layer4 for domain-specific feature extraction
  4. CLAHE contrast enhancement & Stratified Balanced Sampler
==============================================================================
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import get_dataloaders, CLASS_NAMES
from src.models.hybrid_qnn import HybridQNN

CHECKPOINTS_DIR = _ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
HYBRID_CKPT  = CHECKPOINTS_DIR / "best_hybrid_qnn.pt"
CONFIG_PATH  = CHECKPOINTS_DIR / "model_config.json"
HISTORY_PATH = CHECKPOINTS_DIR / "training_history.json"


# ==============================================================================
# Focal Loss
# ==============================================================================

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy loss per sample
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ==============================================================================
# Main Trainer
# ==============================================================================

def train_high_accuracy(epochs: int = 20, n_qubits: int = 4, n_layers: int = 2):
    print("=" * 70)
    print("  HIGH-ACCURACY BALANCED HYBRID QNN TRAINING (v3 - Focal Loss)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, use_cutmix=False)

    # Class counts: Normal: 133, Benign: 437, Malignant: 210
    # Inverse frequency weights: Normal ~ 3.28, Benign ~ 1.0, Malignant ~ 2.08
    counts = np.array([133, 437, 210], dtype=np.float32)
    weights = torch.tensor((counts.max() / counts), dtype=torch.float32).to(device)
    print(f"[*] Class weights (Focal Loss): Normal={weights[0]:.2f}, Benign={weights[1]:.2f}, Malignant={weights[2]:.2f}")

    # Build model (v1 architecture with 4 qubits, 2 layers)
    model = HybridQNN(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=3,
        unfreeze_layer4=True,
        version="v1"
    ).to(device)

    # Also unfreeze layer3 for richer feature representations
    for param in model.backbone.layer3.parameters():
        param.requires_grad = True

    criterion = MultiClassFocalLoss(alpha=weights, gamma=2.0, label_smoothing=0.05)

    # Optimizer with differential learning rates
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-4},
        {"params": head_params,     "lr": 1.2e-3},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = 0.0  # Balanced Metric: (Macro F1 + Balanced Accuracy) / 2
    best_epoch = 0
    history    = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_corr, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_corr += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        scheduler.step()
        tr_acc  = 100.0 * train_corr / train_total
        tr_loss = train_loss / train_total

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_preds   = np.array(val_preds)
        val_targets = np.array(val_targets)

        vl_loss = val_loss / len(val_targets)
        vl_acc  = 100.0 * np.mean(val_preds == val_targets)

        # Calculate per-class recalls
        recalls = recall_score(val_targets, val_preds, average=None, zero_division=0) * 100.0
        n_acc, b_acc, m_acc = recalls[0], recalls[1], recalls[2]
        balanced_acc = np.mean(recalls)
        macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100.0

        # Selection score balances macro F1 and balanced accuracy so all 3 classes are accurate
        combined_score = (balanced_acc + macro_f1) / 2.0

        elapsed = time.time() - t0

        status = ""
        # Require all 3 classes to have > 15% accuracy before saving to avoid single-class collapse
        if combined_score > best_score and min(n_acc, b_acc, m_acc) > 10.0:
            best_score = combined_score
            best_epoch = epoch
            torch.save(model.state_dict(), HYBRID_CKPT)
            status = " <-- BEST SAVED"

        print(
            f"Epoch [{epoch:2d}/{epochs}]  "
            f"Loss: {tr_loss:.4f} | Tr Acc: {tr_acc:.1f}% | "
            f"Val Acc: {vl_acc:.1f}% (Bal Acc: {balanced_acc:.1f}%, F1: {macro_f1:.1f}%) "
            f"[Norm: {n_acc:.0f}%, Benign: {b_acc:.0f}%, Malig: {m_acc:.0f}%] "
            f"({elapsed:.0f}s){status}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 2),
            "val_loss": round(vl_loss, 4),
            "val_acc": round(vl_acc, 2),
            "balanced_acc": round(balanced_acc, 2),
            "macro_f1": round(macro_f1, 2),
            "val_per_class": {
                "Normal": round(n_acc, 2),
                "Benign": round(b_acc, 2),
                "Malignant": round(m_acc, 2)
            }
        })

    # Save model config and training history
    config = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "classes": CLASS_NAMES,
        "unfreeze_layer4": True,
        "version": "v1",
        "best_combined_score": round(best_score, 2),
        "best_epoch": best_epoch
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"  TRAINING COMPLETED | Best Combined Score: {best_score:.2f}% (Epoch {best_epoch})")
    print(f"  Saved to: {HYBRID_CKPT}")
    print("=" * 70)


if __name__ == "__main__":
    train_high_accuracy(epochs=15, n_qubits=4, n_layers=2)
