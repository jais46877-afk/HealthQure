"""
src/train_classical_first.py
==============================================================================
Two-Stage High-Accuracy Hybrid QNN Trainer
==============================================================================
Stage 1: Fine-tunes ResNet-18 backbone on Breast Ultrasound (BUSI) scans
         using Focal Loss and Balanced Weighted Sampler to master feature extraction.
Stage 2: Integrates the Variational Quantum Circuit (QNN) onto the optimized feature
         space to produce high-precision predictions across Normal, Benign, and Malignant.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score

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


def train_two_stage(epochs_stage1: int = 8, epochs_stage2: int = 6):
    print("=" * 70)
    print("  TWO-STAGE HIGH-ACCURACY HYBRID QNN TRAINER")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, use_cutmix=False)

    # Class weights for CrossEntropy: Normal=1.3, Benign=0.8, Malignant=1.2
    weights = torch.tensor([1.3, 0.8, 1.2], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    # Instantiate HybridQNN v1 (4 qubits, 2 layers)
    model = HybridQNN(
        n_qubits=4,
        n_layers=2,
        n_classes=3,
        unfreeze_layer4=True,
        version="v1"
    ).to(device)

    # STAGE 1: Warmup & Fine-tune Backbone + Bottleneck (Freeze QNN temporarily)
    print("\n--- STAGE 1: Fine-tuning ResNet Feature Extractor ---")
    optimizer_st1 = torch.optim.AdamW([
        {"params": model.backbone.layer4.parameters(), "lr": 3e-4},
        {"params": model.bottleneck.parameters(),       "lr": 1.5e-3},
        {"params": model.classifier.parameters(),       "lr": 1.5e-3},
    ], weight_decay=1e-4)

    for epoch in range(1, epochs_stage1 + 1):
        t0 = time.time()
        model.train()
        train_loss, train_corr, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_st1.zero_grad()

            # Classical pass skipping qlayer during Stage 1 warmup
            feats  = model.backbone(imgs)
            angles = model.bottleneck(feats)
            logits = model.classifier(angles)  # shape (B, 3)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_st1.step()

            train_loss += loss.item() * imgs.size(0)
            train_corr += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        tr_acc  = 100.0 * train_corr / train_total
        tr_loss = train_loss / train_total

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats  = model.backbone(imgs)
                angles = model.bottleneck(feats)
                logits = model.classifier(angles)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        recalls = recall_score(val_targets, val_preds, average=None, zero_division=0) * 100.0
        vl_acc  = 100.0 * np.mean(np.array(val_preds) == np.array(val_targets))
        elapsed = time.time() - t0
        print(f"Stage 1 Epoch [{epoch}/{epochs_stage1}]  Loss: {tr_loss:.4f} | Tr Acc: {tr_acc:.1f}% | Val Acc: {vl_acc:.1f}% [Norm: {recalls[0]:.0f}%, Benign: {recalls[1]:.0f}%, Malig: {recalls[2]:.0f}%] ({elapsed:.0f}s)")

    # STAGE 2: Joint Quantum-Classical Fine-Tuning
    print("\n--- STAGE 2: Joint Hybrid Quantum-Classical Training ---")
    optimizer_st2 = torch.optim.AdamW([
        {"params": model.backbone.layer4.parameters(), "lr": 1e-4},
        {"params": model.bottleneck.parameters(),       "lr": 5e-4},
        {"params": model.qlayer.parameters(),           "lr": 2e-3},
        {"params": model.classifier.parameters(),       "lr": 1e-3},
    ], weight_decay=1e-4)

    best_balanced_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs_stage2 + 1):
        t0 = time.time()
        model.train()
        train_loss, train_corr, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_st2.zero_grad()

            # Full hybrid forward pass
            logits = model(imgs)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_st2.step()

            train_loss += loss.item() * imgs.size(0)
            train_corr += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

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
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_epoch = epoch
            torch.save(model.state_dict(), HYBRID_CKPT)
            status = " <-- BEST SAVED"

        print(
            f"Stage 2 Epoch [{epoch}/{epochs_stage2}]  "
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
        "n_qubits": 4,
        "n_layers": 2,
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
    print(f"  TWO-STAGE TRAINING COMPLETED | Best Balanced Acc: {best_balanced_acc:.2f}%")
    print(f"  Saved to: {HYBRID_CKPT}")
    print("=" * 70)


if __name__ == "__main__":
    train_two_stage(epochs_stage1=8, epochs_stage2=6)
