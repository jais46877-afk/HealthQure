"""
src/heart/train.py
==============================================================================
Training Script for Heart Disease Quantum Support Vector Classifier (QSVC)
==============================================================================
"""

import os
import pickle
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

HEART_CHECKPOINT_DIR = _ROOT / "checkpoints" / "heart"
HEART_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PKL = HEART_CHECKPOINT_DIR / "heart_qsvc_model.pkl"
META_JSON = HEART_CHECKPOINT_DIR / "heart_qsvc_metadata.json"

from src.heart.data_loader import generate_chd_dataset, CHDDataPipeline, FEATURE_NAMES
from src.heart.quantum_model import QuantumKernelQSVC


def train_heart_model(n_samples: int = 200, test_size: float = 0.2):
    print("=" * 65)
    print("  TRAINING HEART DISEASE QUANTUM MODEL (QSVC)")
    print("=" * 65)

    df = generate_chd_dataset(n_samples=n_samples, random_state=42)
    pipeline = CHDDataPipeline(n_components=4, random_state=42)

    X_angles = pipeline.fit_transform(df)
    y = df["chd_risk"].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_angles, y, test_size=test_size, random_state=42, stratify=y
    )

    qsvc = QuantumKernelQSVC(n_qubits=4, feature_map_type="zz_feature_map", C=1.5, random_state=42)
    qsvc.fit(X_train, y_train)

    eval_metrics = qsvc.evaluate(X_test, y_test)
    print(f"[*] Training complete!")
    print(f"    Test Accuracy : {eval_metrics['accuracy'] * 100:.2f}%")
    print(f"    F1 Score      : {eval_metrics['f1_score']:.4f}")
    print(f"    ROC AUC       : {eval_metrics['roc_auc']:.4f}")

    # Save artifacts
    artifacts = {
        "pipeline": pipeline,
        "qsvc": qsvc
    }
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(artifacts, f)

    meta = {
        "accuracy": eval_metrics["accuracy"],
        "f1_score": eval_metrics["f1_score"],
        "roc_auc": eval_metrics["roc_auc"],
        "feature_names": FEATURE_NAMES,
        "n_samples": n_samples,
        "n_qubits": 4,
        "feature_map": "zz_feature_map"
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[*] Artifacts saved to: {HEART_CHECKPOINT_DIR}")
    return meta


if __name__ == "__main__":
    train_heart_model()
