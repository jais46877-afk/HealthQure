"""
train.py - Trains the Quantum Support Vector Classifier (QSVC) on CHD Dataset

1. Generates or loads the clinical CHD dataset.
2. Standardizes and preprocesses features into quantum state angles.
3. Computes the PennyLane Quantum State Fidelity Kernel Gram matrix.
4. Fits the precomputed SVM classifier.
5. Computes clinical metrics: Accuracy, F1-Score, ROC-AUC, Confusion Matrix.
6. Serializes model artifacts and metrics to disk for inference.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

from data_loader import CHDDataPipeline, generate_chd_dataset, FEATURE_NAMES, TARGET_NAME
from quantum_model import QuantumKernelQSVC

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def run_training(
    n_samples: int = 150,
    test_size: float = 0.25,
    n_qubits: int = 4,
    feature_map_type: str = "zz_feature_map",
    C: float = 1.0,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes training workflow and saves all necessary artifacts.
    """
    print("=" * 60)
    print(f"Starting Quantum CHD Classifier Training [Qubits: {n_qubits}, Map: {feature_map_type}]")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Dataset Generation / Loading
    print(f"[1/5] Generating clinical CHD dataset (N={n_samples} patients)...")
    df = generate_chd_dataset(n_samples=n_samples, random_state=random_state)
    csv_path = os.path.join(MODEL_DIR, "chd_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"      Saved dataset to {csv_path}")

    # 2. Preprocessing & Scaling
    print("[2/5] Standardizing numeric features & mapping to quantum angle domain [-pi, pi]...")
    pipeline = CHDDataPipeline(n_qubits=n_qubits, use_pca=True, random_state=random_state)
    X_train_q, X_test_q, y_train, y_test = pipeline.fit_transform_train_test(df, test_size=test_size)
    print(f"      Train samples: {len(X_train_q)} | Test samples: {len(X_test_q)}")

    # 3. Quantum Kernel Matrix Computation & Fitting
    print(f"[3/5] Computing PennyLane Quantum State Fidelity Gram matrix ({len(X_train_q)}x{len(X_train_q)})...")
    qsvc = QuantumKernelQSVC(
        n_qubits=n_qubits,
        feature_map_type=feature_map_type,
        C=C,
        random_state=random_state
    )
    K_train = qsvc.fit(X_train_q, y_train)
    print(f"      Gram matrix computed successfully. Support vectors: {len(qsvc.svc.support_)}/{len(X_train_q)}")

    # 4. Model Evaluation
    print("[4/5] Evaluating test set predictions & quantum metrics...")
    metrics = qsvc.evaluate(X_test_q, y_test)
    print(f"      Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"      F1-Score: {metrics['f1_score']:.4f}")
    print(f"      ROC-AUC:  {metrics['roc_auc']:.4f}")

    # Subsample Gram matrix for UI heatmap visualization (first 25 samples)
    sub_k_size = min(25, len(K_train))
    sample_sub_gram = np.round(K_train[:sub_k_size, :sub_k_size], 4).tolist()

    # 5. Saving Artifacts
    print("[5/5] Serializing model state, pipeline, and evaluation metrics...")
    artifacts = {
        "pipeline_state": pipeline.export_state(),
        "X_train_q": X_train_q.tolist(),
        "y_train": y_train.tolist(),
        "support_vectors_indices": qsvc.svc.support_.tolist(),
        "dual_coef": qsvc.svc.dual_coef_.tolist(),
        "intercept": qsvc.svc.intercept_.tolist(),
        "metrics": metrics,
        "sample_gram_matrix": sample_sub_gram,
        "config": {
            "n_qubits": n_qubits,
            "feature_map_type": feature_map_type,
            "C": C,
            "n_samples": n_samples,
            "test_size": test_size,
            "random_state": random_state
        }
    }

    metadata_path = os.path.join(MODEL_DIR, "qsvc_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(artifacts, f, indent=2)

    # Save pickle of fitted QSVC and Pipeline for direct Python runtime
    pkl_path = os.path.join(MODEL_DIR, "qsvc_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "qsvc": qsvc}, f)

    print(f"      Artifacts saved to {MODEL_DIR}/")
    print("=" * 60)
    print("Quantum CHD Model Training Complete!")
    print("=" * 60)
    return artifacts


if __name__ == "__main__":
    run_training()
