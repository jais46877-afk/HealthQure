"""
src/train_perfect_balance.py
==============================================================================
Perfectly Balanced 3-Class Hybrid QNN Trainer
==============================================================================
Optimizes for high recall across ALL THREE classes (Normal, Benign, Malignant).
Ensures no single class is neglected.
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


def train_perfect_balance(epochs: int = 15, n_qubits: int = 4, n_layers: int = 2):
    print("=" * 70)
    print("  PERFECTLY BALANCED 3-CLASS HYBRID QNN TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, use_cutmix=False)

    # Moderate weights to prevent any single class collapse
    # Normal: 133, Benign: 437, Malignant: 210
    weights = torch.tensor([1.35, 0.85, 1.25], dtype=torch.float32).to(device)

    model = HybridQNN(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=3,
        unfreeze_layer4=True,
        version="v1"
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.08)

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 2e-4},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": 1.5e-3},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_balanced_acc = 0.0
    best_epoch = 0
    history = []

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

        recalls = recall_score(val_targets, val_preds, average=None, zero_division=0) * 100.0
        n_acc, b_acc, m_acc = recalls[0], recalls[1], recalls[2]
        balanced_acc = np.mean(recalls)
        macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100.0

        elapsed = time.time() - t0

        status = ""
        # Ensure all 3 classes have at least 25% recall so no class is collapsed
        min_class_recall = min(n_acc, b_acc, m_acc)
        if balanced_acc > best_balanced_acc and min_class_recall >= 25.0:
            best_balanced_acc = balanced_acc
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

    config = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "classes": CLASS_NAMES,
        "unfreeze_layer4": True,
        "version": "v1",
        "best_balanced_acc": round(best_balanced_acc, 2),
        "best_epoch": best_epoch
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"  TRAINING COMPLETED | Best Balanced Accuracy: {best_balanced_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Saved to: {HYBRID_CKPT}")
    print("=" * 70)


if __name__ == "__main__":
    train_perfect_balance(epochs=12, n_qubits=4, n_layers=2)
