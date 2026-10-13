"""
src/evaluate.py
==============================================================================
Hybrid QNN Evaluation & Benchmarking  (QNN ONLY)
==============================================================================

Reads ./checkpoints/model_config.json, reconstructs the exact architecture,
evaluates on the 15% test split, and exports metrics to benchmark_results.json

Metrics:
  - Trainable parameters
  - Overall accuracy (%)
  - Macro F1-score
  - Per-class Precision / Recall / F1
  - Malignant-class sensitivity (recall)
  - Per-sample inference latency (ms)

Usage:
    python src/evaluate.py
==============================================================================
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---- Path bootstrap ---------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import get_dataloaders, CLASS_NAMES
from src.models.hybrid_qnn import HybridQNN

# ==============================================================================
# Paths
# ==============================================================================

CHECKPOINTS_DIR = _ROOT / "checkpoints"
CONFIG_PATH     = CHECKPOINTS_DIR / "model_config.json"
HYBRID_CKPT     = CHECKPOINTS_DIR / "best_hybrid_qnn.pt"
RESULTS_PATH    = CHECKPOINTS_DIR / "benchmark_results.json"
MALIGNANT_IDX   = 2


# ==============================================================================
# Inference
# ==============================================================================

@torch.no_grad()
def _run_inference(model, loader, device) -> tuple:
    """Return (y_true, y_pred, latency_ms_per_sample)."""
    model.eval()
    model.to(device)
    all_true, all_pred = [], []
    total_time = 0.0
    n_samples  = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        t0   = time.perf_counter()
        logits = model(imgs)
        total_time += time.perf_counter() - t0
        n_samples  += imgs.size(0)
        all_true.extend(labels.numpy())
        all_pred.extend(logits.argmax(1).cpu().numpy())

    latency_ms = (total_time / n_samples) * 1000.0
    return np.array(all_true), np.array(all_pred), latency_ms


# ==============================================================================
# Metrics
# ==============================================================================

def _compute_metrics(y_true, y_pred, latency_ms, model) -> dict:
    acc      = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    tp_mal = cm[MALIGNANT_IDX, MALIGNANT_IDX]
    fn_mal = cm[MALIGNANT_IDX, :].sum() - tp_mal
    mal_sensitivity = float(tp_mal / (tp_mal + fn_mal)) if (tp_mal + fn_mal) > 0 else 0.0

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    return {
        "trainable_params":       trainable,
        "total_params":           total,
        "accuracy_pct":           round(float(acc), 4),
        "macro_f1":               round(float(macro_f1), 4),
        "malignant_sensitivity":  round(float(mal_sensitivity), 4),
        "latency_ms_per_sample":  round(float(latency_ms), 4),
    }


def _print_results(metrics: dict, n_qubits: int, n_layers: int) -> None:
    w = 56
    print("\n" + "=" * w)
    print("  HYBRID QNN  -  TEST SET BENCHMARK  (15%)")
    print("=" * w)
    rows = [
        ("Qubits",                   n_qubits),
        ("Circuit depth (layers)",   n_layers),
        ("Trainable parameters",     "{:,}".format(metrics["trainable_params"])),
        ("Total parameters",         "{:,}".format(metrics["total_params"])),
        ("Accuracy (%)",             "{:.2f}%".format(metrics["accuracy_pct"])),
        ("Macro F1-Score",           "{:.4f}".format(metrics["macro_f1"])),
        ("Malignant Sensitivity",    "{:.4f}".format(metrics["malignant_sensitivity"])),
        ("Latency (ms / sample)",    "{:.2f} ms".format(metrics["latency_ms_per_sample"])),
    ]
    for label, val in rows:
        print("  {:<28} {}".format(label, val))
    print("=" * w)


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    # -- Load config -----------------------------------------------------------
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "model_config.json not found. Run `python src/train.py` first."
        )
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    n_qubits = cfg["n_qubits"]
    n_layers = cfg["n_layers"]

    print("\n[*] Config loaded: n_qubits={}, n_layers={}".format(n_qubits, n_layers))

    # -- Device ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[*] Device: {}".format(device))

    # -- Test data -------------------------------------------------------------
    print("[*] Loading test split ...")
    _, _, test_loader = get_dataloaders(batch_size=32, verbose=False)

    # -- Load model ------------------------------------------------------------
    print("[*] Loading Hybrid QNN weights ...")
    if not HYBRID_CKPT.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(HYBRID_CKPT))

    version = cfg.get("version", "v1")
    model = HybridQNN(n_qubits=n_qubits, n_layers=n_layers, version=version)
    model.load_state_dict(torch.load(HYBRID_CKPT, map_location=device))

    # -- Evaluate --------------------------------------------------------------
    print("[*] Running inference on test set ...")
    y_true, y_pred, latency_ms = _run_inference(model, test_loader, device)

    # -- Metrics ---------------------------------------------------------------
    metrics = _compute_metrics(y_true, y_pred, latency_ms, model)
    _print_results(metrics, n_qubits, n_layers)

    # -- Per-class report ------------------------------------------------------
    print("\n-- Per-class Classification Report --")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    # -- Confusion matrix ------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("-- Confusion Matrix (rows=actual, cols=predicted) --")
    header = "         " + "  ".join("{:>10}".format(c) for c in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row_str = "{:>9}".format(row_name) + "  ".join(
            "{:>10}".format(cm[i, j]) for j in range(3)
        )
        print(row_str)

    # -- Export ----------------------------------------------------------------
    export = {"config": cfg, "metrics": metrics}
    with open(RESULTS_PATH, "w") as f:
        json.dump(export, f, indent=2)
    print("\n[*] Results saved -> {}".format(RESULTS_PATH))


if __name__ == "__main__":
    main()
