"""
src/train.py
==============================================================================
Hybrid QNN Training Pipeline  — v2 (Accuracy Boost Edition)
==============================================================================
Changes vs v1:
  1. Focal Loss (gamma=2.0) — focuses harder on misclassified hard samples
  2. AdamW with warm-up (10% of epochs) + CosineAnnealingWarmRestarts
  3. Gradient clipping (max_norm=1.0) to stabilize quantum gradients
  4. Dual learning rates: layer4 at 1e-5, rest at cfg["lr"]
  5. Default: epochs=50, qubits=6, layers=3, batch=32
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import get_dataloaders, CLASS_NAMES
from src.models.hybrid_qnn import HybridQNN


# ==============================================================================
# Constants
# ==============================================================================

CHECKPOINTS_DIR = _ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

HYBRID_CKPT = CHECKPOINTS_DIR / "best_hybrid_qnn.pt"
CONFIG_PATH = CHECKPOINTS_DIR / "model_config.json"

VALID_QUBITS = {2, 4, 6, 8}

DEFAULT_EPOCHS     = 50
DEFAULT_QUBITS     = 6
DEFAULT_LAYERS     = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 0.001


# ==============================================================================
# Focal Loss
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma  : float  Focusing parameter. Higher = more focus on hard samples.
                    gamma=0 → standard cross-entropy.
    alpha  : float or None  Per-class weight scalar (None = uniform).
    reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.reduction       = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        # Cross entropy with label smoothing
        ce_loss = F.cross_entropy(
            logits, targets,
            weight      = self.alpha.to(logits.device) if self.alpha is not None else None,
            reduction   = "none",
            label_smoothing = self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)                          # probability of correct class
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss  # focal modulation

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_focal_loss(train_labels: np.ndarray | None = None) -> FocalLoss:
    """Build Focal Loss with optional inverse-frequency alpha weights."""
    alpha = None
    if train_labels is not None:
        counts = np.bincount(train_labels, minlength=3).astype(float)
        weights = 1.0 / np.where(counts == 0, 1.0, counts)
        weights = weights / weights.sum() * len(weights)   # normalize
        alpha = torch.tensor(weights, dtype=torch.float32)
        print("  [Loss] FocalLoss(gamma=2.0)  class weights: {}".format(
            ["{:.3f}".format(w) for w in weights]
        ))

    return FocalLoss(gamma=2.0, alpha=alpha, label_smoothing=0.05)


# ==============================================================================
# UI helpers
# ==============================================================================

def _banner(title: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print("  " + title)
    print("=" * width)


def _prompt(msg: str, default, choices=None, cast=int):
    while True:
        raw = input(msg).strip()
        if raw == "":
            return default
        try:
            value = cast(raw)
        except ValueError:
            print("  [!] Expected a {}. Try again.".format(cast.__name__))
            continue
        if choices and value not in choices:
            print("  [!] Choose from: {}".format(sorted(choices)))
            continue
        return value


def _print_config(cfg: dict, device: str) -> None:
    _banner("HQNN TRAINING CONFIGURATION  v2")
    rows = [
        ("Epochs",           cfg["epochs"]),
        ("Qubits",           cfg["n_qubits"]),
        ("Circuit depth",    cfg["n_layers"]),
        ("Batch size",       cfg["batch_size"]),
        ("Learning rate",    cfg["lr"]),
        ("Device",           device.upper()),
        ("Loss",             "FocalLoss(gamma=2.0, label_smoothing=0.05)"),
        ("Sampler",          "WeightedRandomSampler + CutMix"),
        ("Scheduler",        "Warm-up (10%) + CosineAnnealingWarmRestarts"),
        ("Grad clipping",    "max_norm=1.0"),
        ("Quantum",          "Data Re-Uploading | Dual Pauli | Layer4 Unfrozen"),
        ("Input size",       "256x256 with CLAHE"),
    ]
    for k, v in rows:
        print("  {:<22} {}".format(k, v))
    print("=" * 60)


# ==============================================================================
# Hyperparameter setup
# ==============================================================================

def interactive_setup(args: argparse.Namespace) -> dict:
    has_cli = any(v is not None for v in vars(args).values()
                  if not isinstance(v, bool))

    if not has_cli:
        _banner("HQNN BREAST CANCER  v2  —  HYPERPARAMETER SETUP")
        epochs     = _prompt("[?] Epochs         (default {}): ".format(DEFAULT_EPOCHS),
                             default=DEFAULT_EPOCHS, cast=int)
        n_qubits   = _prompt("[?] Qubits         (default {}, choices 2/4/6/8): ".format(DEFAULT_QUBITS),
                             default=DEFAULT_QUBITS, choices=VALID_QUBITS, cast=int)
        n_layers   = _prompt("[?] Circuit depth  (default {}): ".format(DEFAULT_LAYERS),
                             default=DEFAULT_LAYERS, cast=int)
        batch_size = _prompt("[?] Batch size     (default {}): ".format(DEFAULT_BATCH_SIZE),
                             default=DEFAULT_BATCH_SIZE, cast=int)
        lr         = _prompt("[?] Learning rate  (default {}): ".format(DEFAULT_LR),
                             default=DEFAULT_LR, cast=float)
    else:
        epochs     = args.epochs if args.epochs is not None else DEFAULT_EPOCHS
        n_qubits   = args.qubits if args.qubits is not None else DEFAULT_QUBITS
        n_layers   = args.layers if args.layers is not None else DEFAULT_LAYERS
        batch_size = args.batch  if args.batch  is not None else DEFAULT_BATCH_SIZE
        lr         = args.lr     if args.lr     is not None else DEFAULT_LR

        if n_qubits not in VALID_QUBITS:
            print("  [WARNING] n_qubits={} invalid. Defaulting to {}.".format(
                n_qubits, DEFAULT_QUBITS))
            n_qubits = DEFAULT_QUBITS

    return {"epochs": epochs, "n_qubits": n_qubits,
            "n_layers": n_layers, "batch_size": batch_size, "lr": lr}


# ==============================================================================
# Training loop
# ==============================================================================

def _train_one_epoch(model, loader, criterion, optimizer, device, epoch, total):
    model.train()
    total_loss = correct = n = 0
    pbar = tqdm(loader,
                desc="  Epoch [{:3d}/{}] train".format(epoch, total),
                leave=False, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        # Gradient clipping — stabilizes quantum parameter updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
        pbar.set_postfix(loss="{:.4f}".format(loss.item()))
    return total_loss / n, 100.0 * correct / n


@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = n = 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            per_class[name] = 100.0 * (np.array(all_preds)[mask] == i).sum() / mask.sum()
        else:
            per_class[name] = 0.0
    return total_loss / n, 100.0 * correct / n, per_class


# ==============================================================================
# Warm-up LR scheduler wrapper
# ==============================================================================

class WarmupCosineScheduler:
    """
    Linear warm-up for the first warmup_epochs epochs,
    then CosineAnnealingWarmRestarts for the rest.
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr
        self.cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, (total_epochs - warmup_epochs) // 2), T_mult=1, eta_min=1e-6
        )
        self._epoch = 0

    def step(self):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            lr = self.base_lr * (self._epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.cosine_sched.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ==============================================================================
# Main training function
# ==============================================================================

def train_hybrid_qnn(model, train_loader, val_loader, cfg, device,
                     train_labels: np.ndarray | None = None):
    _banner("TRAINING  →  Hybrid QNN v2  (Data Re-Uploading)")
    p = model.count_parameters()
    print("  Trainable params : {:,}".format(p["trainable"]))
    print("  Total params     : {:,}".format(p["total"]))
    print("  Qubits: {}   Layers: {}   Q-out: {}".format(
        model.n_qubits, model.n_layers, 2 * model.n_qubits))

    model     = model.to(device)
    criterion = get_focal_loss(train_labels)

    # Dual LR: backbone layer4 uses much smaller LR
    backbone_layer4_params = [
        p for p in model.backbone.layer4.parameters() if p.requires_grad
    ] if model.unfreeze_layer4 else []
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "backbone.layer4" not in n
    ]

    param_groups = [
        {"params": other_params,          "lr": cfg["lr"]},
        {"params": backbone_layer4_params, "lr": cfg["lr"] * 0.01},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Warm-up (10%) + Cosine restart
    warmup_epochs = max(1, cfg["epochs"] // 10)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=warmup_epochs,
        total_epochs=cfg["epochs"], base_lr=cfg["lr"]
    )

    best_val_acc = -1.0
    best_epoch   = 0
    history      = []

    print()
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc          = _train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, cfg["epochs"])
        vl_loss, vl_acc, per_cls = _validate(
            model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        status  = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), HYBRID_CKPT)
            status = "  <-- BEST SAVED"

        cls_str = "  ".join(
            "{}:{:.0f}%".format(n[0], v)
            for n, v in per_cls.items()
        )
        print(
            "  Epoch [{:3d}/{}]  "
            "tr_loss={:.4f}  tr_acc={:.1f}%  "
            "vl_acc={:.1f}%  [{}]  "
            "({:.0f}s){}".format(
                epoch, cfg["epochs"],
                tr_loss, tr_acc,
                vl_acc, cls_str,
                elapsed, status,
            )
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 2),
            "val_loss": round(vl_loss, 4),   "val_acc":   round(vl_acc, 2),
            "val_per_class": {k: round(v, 2) for k, v in per_cls.items()},
        })

    model.load_state_dict(torch.load(HYBRID_CKPT, map_location=device))

    _banner("TRAINING COMPLETE")
    print("  Best val accuracy : {:.2f}%  (epoch {})".format(best_val_acc, best_epoch))
    print("  Checkpoint        : {}".format(HYBRID_CKPT))
    print("=" * 60 + "\n")

    with open(CHECKPOINTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model


# ==============================================================================
# CLI
# ==============================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Train Hybrid QNN v2 for Breast Cancer Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int,   default=None)
    p.add_argument("--qubits", type=int,   default=None)
    p.add_argument("--layers", type=int,   default=None)
    p.add_argument("--batch",  type=int,   default=None)
    p.add_argument("--lr",     type=float, default=None)
    p.add_argument("--no-interactive", action="store_true")
    return p


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    cfg        = interactive_setup(args)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)

    _print_config(cfg, device_str)

    if not args.no_interactive:
        ans = input("\nProceed with training? [Y/n]: ").strip().lower()
        if ans in {"n", "no"}:
            print("Aborted."); sys.exit(0)

    print("\n[*] Loading dataset (CLAHE + CutMix) ...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=cfg["batch_size"], use_cutmix=True
    )

    # Collect training labels for Focal Loss alpha weights
    all_train_labels = []
    for _, lbls in train_loader:
        all_train_labels.extend(lbls.numpy())
    train_labels_arr = np.array(all_train_labels)

    model_config = {
        "n_qubits":       cfg["n_qubits"],
        "n_layers":       cfg["n_layers"],
        "classes":        CLASS_NAMES,
        "unfreeze_layer4": True,
        "version":        "v2",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(model_config, f, indent=2)
    print("[*] Config saved → {}".format(CONFIG_PATH))

    print("\n[*] Building Hybrid QNN v2 ...")
    model = HybridQNN(
        n_qubits        = cfg["n_qubits"],
        n_layers        = cfg["n_layers"],
        unfreeze_layer4 = True,
    )
    print(model)

    train_hybrid_qnn(model, train_loader, val_loader, cfg, device, train_labels_arr)

    print("  Next steps:")
    print("    Evaluate : python src/evaluate.py")
    print("    Web app  : uvicorn api.main:app --port 8000\n")


if __name__ == "__main__":
    main()
