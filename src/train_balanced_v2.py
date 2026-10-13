"""
src/train_balanced_v2.py
==============================================================================
Balanced High-Accuracy Hybrid QNN Trainer (v2 - Equalized Boundaries)
==============================================================================
Presents balanced decision boundaries so that Normal, Benign, and Malignant
are all predicted with high precision and recall.
==============================================================================
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import get_dataloaders, CLASS_NAMES
from src.models.hybrid_qnn import HybridQNN

CHECKPOINTS_DIR = _ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
HYBRID_CKPT = CHECKPOINTS_DIR / "best_hybrid_qnn.pt"
CONFIG_PATH = CHECKPOINTS_DIR / "model_config.json"


def train_balanced_v2(epochs: int = 15, n_qubits: int = 4, n_layers: int = 2):
    print("=" * 65)
    print("  BALANCED HIGH-ACCURACY TRAINING v2  (3-Class Equalized)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, use_cutmix=False)

    # Moderate weights to balance precision & recall for all 3 classes
    weights = torch.tensor([1.1, 0.85, 1.15], dtype=torch.float32).to(device)

    model = HybridQNN(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=3,
        unfreeze_layer4=True,
        version="v1"
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    # Optimizer: backbone layer4 LR=2e-4, head LR=1e-3
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 2e-4},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": 1.5e-3},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_epoch = 0

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
        tr_acc = 100.0 * train_corr / train_total
        tr_loss = train_loss / train_total

        # Validation evaluation
        model.eval()
        val_loss, val_corr, val_total = 0.0, 0, 0
        per_class_corr = [0, 0, 0]
        per_class_tot  = [0, 0, 0]

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                val_corr += (preds == labels).sum().item()
                val_total += imgs.size(0)

                for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                    per_class_tot[l] += 1
                    if p == l:
                        per_class_corr[l] += 1

        vl_acc = 100.0 * val_corr / val_total
        vl_loss = val_loss / val_total
        elapsed = time.time() - t0

        n_acc = 100.0 * per_class_corr[0] / max(1, per_class_tot[0])
        b_acc = 100.0 * per_class_corr[1] / max(1, per_class_tot[1])
        m_acc = 100.0 * per_class_corr[2] / max(1, per_class_tot[2])

        status = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            torch.save(model.state_dict(), HYBRID_CKPT)
            status = " <-- BEST SAVED"

        print(
            f"Epoch [{epoch:2d}/{epochs}]  "
            f"Loss: {tr_loss:.4f} | Tr Acc: {tr_acc:.1f}% | "
            f"Val Acc: {vl_acc:.1f}%  "
            f"[Normal: {n_acc:.0f}%, Benign: {b_acc:.0f}%, Malignant: {m_acc:.0f}%] "
            f"({elapsed:.0f}s){status}"
        )

    # Save model config
    config = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "classes": CLASS_NAMES,
        "unfreeze_layer4": True,
        "version": "v1",
        "best_val_acc": round(best_val_acc, 2)
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 65)
    print(f"  TRAINING COMPLETED | Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Saved to: {HYBRID_CKPT}")
    print("=" * 65)


if __name__ == "__main__":
    train_balanced_v2(epochs=12, n_qubits=4, n_layers=2)
