"""
src/train_vqc_hybrid.py
==============================================================================
HYBRID QUANTUM NEURAL NETWORK (VQC) TRAINER
==============================================================================
Trains a 4-qubit Hybrid Quantum-Classical model for Breast Ultrasound diagnosis.
Saves independently to checkpoints/hybrid_vqc_model.pt without touching the
classical model.

Architecture:
  - Classical Backbone: ResNet-18 feature extractor (512-dim)
  - Quantum Bottleneck: Linear + Tanh mapping to 4 rotation angles [-pi, pi]
  - Variational Quantum Circuit (VQC): 4 qubits, 2 StronglyEntanglingLayers
    measuring Pauli-Z expectations on all 4 wires via PennyLane
  - Classical Head: Linear layer mapping 4 quantum expectation values to 3 class logits
==============================================================================
"""

import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import (
    BreastUltrasoundDataset,
    _build_train_transform,
    _build_eval_transform,
    CLASS_NAMES,
)
from src.models.hybrid_qnn import HybridQNN

CHECKPOINTS_DIR = _ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
VQC_CKPT        = CHECKPOINTS_DIR / "hybrid_vqc_model.pt"
VQC_CONFIG_PATH = CHECKPOINTS_DIR / "vqc_model_config.json"
VQC_HIST_PATH   = CHECKPOINTS_DIR / "vqc_training_history.json"
VQC_BENCH_PATH  = CHECKPOINTS_DIR / "vqc_benchmark_results.json"


# ==============================================================================
# Focal Loss with Class Weights
# ==============================================================================

class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ==============================================================================
# Sampler with Benign Oversampling
# ==============================================================================

def make_extreme_benign_sampler(labels: np.ndarray, benign_multiplier: float = 3.5):
    counts = np.bincount(labels, minlength=3).astype(float)
    base_weights = 1.0 / np.where(counts == 0, 1.0, counts)
    base_weights[1] *= benign_multiplier
    sample_weights = base_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True,
    )


# ==============================================================================
# Main Training Function
# ==============================================================================

def train_vqc_hybrid(
    epochs_warmup: int = 5,
    epochs_finetune: int = 5,
    batch_size: int = 16,
):
    print("=" * 70)
    print("  4-QUBIT HYBRID QUANTUM NEURAL NETWORK (VQC) TRAINER")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # 1. Dataset & Stratified Splits
    base_ds = BreastUltrasoundDataset(transform=_build_eval_transform(), verbose=True)
    labels  = base_ds.get_labels()
    n_total = len(base_ds)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(splitter.split(np.zeros(n_total), labels))
    temp_labels = labels[temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_local, test_local = next(splitter2.split(np.zeros(len(temp_idx)), temp_labels))
    val_idx  = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    train_labels = labels[train_idx]
    counts = np.bincount(train_labels, minlength=3)
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"  Train class dist: Normal={counts[0]}, Benign={counts[1]}, Malignant={counts[2]}\n")

    train_ds = BreastUltrasoundDataset(transform=_build_train_transform(), verbose=False)
    eval_ds  = BreastUltrasoundDataset(transform=_build_eval_transform(),  verbose=False)

    sampler = make_extreme_benign_sampler(train_labels, benign_multiplier=3.5)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 2. Instantiate 4-Qubit HybridQNN
    # version='v1' uses 4 qubits, 2 layers, and AngleEmbedding + StronglyEntanglingLayers
    model = HybridQNN(
        n_qubits=4,
        n_layers=2,
        n_classes=3,
        unfreeze_layer4=True,
        version="v1",
    ).to(device)

    # Freeze entire backbone during warmup
    for p in model.backbone.parameters():
        p.requires_grad = False

    alpha = torch.tensor([1.0, 2.5, 1.5], device=device, dtype=torch.float32)
    criterion = FocalLossMultiClass(alpha=alpha, gamma=2.5)

    best_balanced_acc = 0.0
    best_epoch = 0
    history = []

    def evaluate_model(loader):
        model.eval()
        total_loss, all_preds, all_lbls = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                total_loss += loss.item() * imgs.size(0)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_lbls.extend(lbls.cpu().numpy())
        n = len(all_lbls)
        avg_loss = total_loss / n
        acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_lbls))
        bal_acc = balanced_accuracy_score(all_lbls, all_preds) * 100.0

        per_cls = {}
        for i, name in enumerate(CLASS_NAMES):
            mask = np.array(all_lbls) == i
            if mask.sum() > 0:
                per_cls[name] = 100.0 * (np.array(all_preds)[mask] == i).sum() / mask.sum()
            else:
                per_cls[name] = 0.0
        return avg_loss, acc, bal_acc, per_cls

    # ==========================================================================
    # PHASE 1: Warmup Quantum Circuit + Bottleneck + Classifier (Backbone frozen)
    # ==========================================================================
    print("--- PHASE 1: Warmup Quantum Circuit & Heads (5 Epochs) ---")
    optimizer_w = torch.optim.AdamW([
        {"params": model.bottleneck.parameters(), "lr": 1.5e-3},
        {"params": model.qlayer.parameters(),     "lr": 2.5e-3},
        {"params": model.classifier.parameters(), "lr": 1.5e-3},
    ], weight_decay=1e-4)

    for ep in range(1, epochs_warmup + 1):
        t0 = time.time()
        model.train()
        train_loss, train_corr, train_total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer_w.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer_w.step()

            train_loss += loss.item() * imgs.size(0)
            train_corr += (logits.argmax(1) == lbls).sum().item()
            train_total += imgs.size(0)

        tr_loss = train_loss / train_total
        tr_acc  = 100.0 * train_corr / train_total

        vl_loss, vl_acc, vl_bal, vl_cls = evaluate_model(val_loader)
        elapsed = time.time() - t0

        tag = ""
        if vl_bal > best_balanced_acc:
            best_balanced_acc = vl_bal
            best_epoch = ep
            torch.save(model.state_dict(), VQC_CKPT)
            tag = " <-- BEST VQC SAVED"

        print(
            f"VQC Ph1 Ep [{ep:02d}/{epochs_warmup}]  "
            f"Loss: {tr_loss:.4f} | TrAcc: {tr_acc:.1f}% | "
            f"Val Bal: {vl_bal:.1f}% "
            f"[N:{vl_cls['Normal']:.0f}% B:{vl_cls['Benign']:.0f}% M:{vl_cls['Malignant']:.0f}%] "
            f"({elapsed:.0f}s){tag}"
        )
        history.append({
            "phase": 1, "epoch": ep,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 2),
            "val_loss": round(vl_loss, 4), "val_acc": round(vl_acc, 2),
            "balanced_acc": round(vl_bal, 2),
            "val_per_class": {k: round(v, 2) for k, v in vl_cls.items()},
        })

    # ==========================================================================
    # PHASE 2: Joint Quantum-Classical Fine-Tuning (Unfreeze Layer 4)
    # ==========================================================================
    print("\n--- PHASE 2: Joint Quantum-Classical Fine-Tuning (Unfreeze Layer 4) ---")
    for p in model.backbone.layer4.parameters():
        p.requires_grad = True

    optimizer_ft = torch.optim.AdamW([
        {"params": model.backbone.layer4.parameters(), "lr": 1.0e-4},
        {"params": model.bottleneck.parameters(),       "lr": 6.0e-4},
        {"params": model.qlayer.parameters(),           "lr": 1.5e-3},
        {"params": model.classifier.parameters(),       "lr": 8.0e-4},
    ], weight_decay=1e-4)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=epochs_finetune, eta_min=1e-5)

    for ep in range(1, epochs_finetune + 1):
        t0 = time.time()
        model.train()
        train_loss, train_corr, train_total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer_ft.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer_ft.step()

            train_loss += loss.item() * imgs.size(0)
            train_corr += (logits.argmax(1) == lbls).sum().item()
            train_total += imgs.size(0)

        scheduler_ft.step()
        tr_loss = train_loss / train_total
        tr_acc  = 100.0 * train_corr / train_total

        vl_loss, vl_acc, vl_bal, vl_cls = evaluate_model(val_loader)
        elapsed = time.time() - t0
        global_ep = epochs_warmup + ep

        tag = ""
        if vl_bal > best_balanced_acc:
            best_balanced_acc = vl_bal
            best_epoch = global_ep
            torch.save(model.state_dict(), VQC_CKPT)
            tag = " <-- BEST VQC SAVED"

        print(
            f"VQC Ph2 Ep [{ep:02d}/{epochs_finetune}]  "
            f"Loss: {tr_loss:.4f} | TrAcc: {tr_acc:.1f}% | "
            f"Val Bal: {vl_bal:.1f}% "
            f"[N:{vl_cls['Normal']:.0f}% B:{vl_cls['Benign']:.0f}% M:{vl_cls['Malignant']:.0f}%] "
            f"({elapsed:.0f}s){tag}"
        )
        history.append({
            "phase": 2, "epoch": global_ep,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 2),
            "val_loss": round(vl_loss, 4), "val_acc": round(vl_acc, 2),
            "balanced_acc": round(vl_bal, 2),
            "val_per_class": {k: round(v, 2) for k, v in vl_cls.items()},
        })

    # ==========================================================================
    # Final Test Set Evaluation
    # ==========================================================================
    print("\n--- FINAL TEST SET EVALUATION (VQC) ---")
    if VQC_CKPT.exists():
        model.load_state_dict(torch.load(VQC_CKPT, map_location=device, weights_only=False))
    model.eval()

    test_loss, test_acc, test_bal, test_cls = evaluate_model(test_loader)
    print(f"  VQC Test Accuracy   : {test_acc:.2f}%")
    print(f"  VQC Balanced Acc    : {test_bal:.2f}%")
    for name, rec in test_cls.items():
        print(f"    - {name:10}: {rec:.2f}% recall")

    # Save VQC configuration
    vqc_config = {
        "n_qubits": 4,
        "n_layers": 2,
        "classes": CLASS_NAMES,
        "unfreeze_layer4": True,
        "version": "v1",
        "model_type": "vqc",
        "best_balanced_acc": round(best_balanced_acc, 2),
        "best_epoch": best_epoch,
        "test_accuracy": round(test_acc, 2),
        "test_balanced_acc": round(test_bal, 2),
        "test_per_class": test_cls,
    }
    with open(VQC_CONFIG_PATH, "w") as f:
        json.dump(vqc_config, f, indent=2)

    with open(VQC_HIST_PATH, "w") as f:
        json.dump(history, f, indent=2)

    vqc_benchmark = {
        "config": vqc_config,
        "metrics": {
            "accuracy_pct": round(test_acc, 4),
            "balanced_acc_pct": round(test_bal, 4),
            "per_class_recall_pct": test_cls,
            "best_val_balanced_acc": round(best_balanced_acc, 2),
            "circuit": "4-Qubit Strongly Entangled Variational Quantum Circuit (PennyLane)",
        }
    }
    with open(VQC_BENCH_PATH, "w") as f:
        json.dump(vqc_benchmark, f, indent=2)

    print("=" * 70)
    print(f"  VQC TRAINING COMPLETE | Best Val Bal Acc: {best_balanced_acc:.2f}%")
    print(f"  Test Balanced Acc: {test_bal:.2f}% | Test Accuracy: {test_acc:.2f}%")
    print(f"  Checkpoint: {VQC_CKPT}")
    print("=" * 70)


if __name__ == "__main__":
    train_vqc_hybrid(epochs_warmup=5, epochs_finetune=5, batch_size=16)
