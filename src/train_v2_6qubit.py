"""
src/train_v2_6qubit.py
==============================================================================
6-Qubit Data Re-Uploading Hybrid QNN Trainer (v2 Architecture)
==============================================================================
Expands Hilbert space to 2⁶ = 64 dimensions with 6 Qubits, 3 Circuit Layers,
Data Re-Uploading, and Dual Pauli (Z + X) measurements (12 features).
Eliminates feature bottleneck and provides clean separation between
Normal, Benign, and Malignant breast ultrasound scans.
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


def train_v2_6qubits(epochs: int = 10, n_qubits: int = 6, n_layers: int = 3):
    print("=" * 70)
    print("  6-QUBIT RE-UPLOADING HYBRID QNN TRAINING (v2 Architecture)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, use_cutmix=False)

    # Class weights for CrossEntropy: Normal=1.2, Benign=0.9, Malignant=1.1
    weights = torch.tensor([1.2, 0.9, 1.1], dtype=torch.float32).to(device)

    model = HybridQNN(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=3,
        unfreeze_layer4=True,
        version="v2"
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1.5e-4},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": 1e-3},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = 0.0
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
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        vl_loss = val_loss / len(all_labels)
        vl_acc  = 100.0 * np.mean(all_preds == all_labels)

        per_cls = {}
        for i, name in enumerate(CLASS_NAMES):
            mask = all_labels == i
            per_cls[name] = 100.0 * (all_preds[mask] == i).sum() / max(1, mask.sum())

        n_acc, b_acc, m_acc = per_cls["Normal"], per_cls["Benign"], per_cls["Malignant"]
        balanced_acc = (n_acc + b_acc + m_acc) / 3.0

        elapsed = time.time() - t0

        status = ""
        # Require all 3 classes to have > 10% recall so no class is dropped
        if balanced_acc > best_score and min(n_acc, b_acc, m_acc) >= 10.0:
            best_score = balanced_acc
            best_epoch = epoch
            torch.save(model.state_dict(), HYBRID_CKPT)
            status = " <-- BEST BALANCED SAVED"

        print(
            f"Epoch [{epoch:2d}/{epochs}]  "
            f"Loss: {tr_loss:.4f} | Tr Acc: {tr_acc:.1f}% | "
            f"Val Acc: {vl_acc:.1f}% (Bal Acc: {balanced_acc:.1f}%) "
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
            "val_per_class": {k: round(v, 2) for k, v in per_cls.items()}
        })

    config = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "classes": CLASS_NAMES,
        "unfreeze_layer4": True,
        "version": "v2",
        "best_balanced_acc": round(best_score, 2),
        "best_epoch": best_epoch
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"  v2 TRAINING COMPLETED | Best Balanced Acc: {best_score:.2f}% (Epoch {best_epoch})")
    print(f"  Saved to: {HYBRID_CKPT}")
    print("=" * 70)


if __name__ == "__main__":
    train_v2_6qubits(epochs=8, n_qubits=6, n_layers=3)
