"""
src/heart/quantum_model.py
==============================================================================
Quantum Support Vector Classifier (QSVC) using PennyLane and PyTorch
==============================================================================
Implements:
1. PennyLane Quantum Circuit with AngleEmbedding and ZZFeatureMap (entanglement).
2. Quantum State Fidelity Kernel: K(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2.
3. PyTorch tensor accelerated quantum state operations.
4. Scikit-Learn Precomputed Kernel Support Vector Classifier (QSVC) with Platt probability calibration.
==============================================================================
"""

import numpy as np
import torch
from typing import Literal, Optional, Tuple, Dict, Any
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

FeatureMapType = Literal["angle", "zz_feature_map"]

N_QUBITS_DEFAULT = 4
DEV = qml.device("default.qubit", wires=N_QUBITS_DEFAULT)


@qml.qnode(DEV, interface="torch")
def qnode_state_angle(x):
    """Encodes 4 features into single-qubit Ry rotations."""
    for i in range(4):
        qml.RY(x[i], wires=i)
    return qml.state()


@qml.qnode(DEV, interface="torch")
def qnode_state_zz(x):
    """
    Havlíček ZZ-Feature Map embedding:
    Hadamard layer + RZ(2*x_i) + CNOT entangling layers with non-linear phase RZ(2*(pi-x_i)*(pi-x_j)).
    """
    for i in range(4):
        qml.Hadamard(wires=i)
    for i in range(4):
        qml.RZ(2.0 * x[i], wires=i)
    for i in range(4):
        j = (i + 1) % 4
        phi = 2.0 * (torch.pi - x[i]) * (torch.pi - x[j])
        qml.CNOT(wires=[i, j])
        qml.RZ(phi, wires=j)
        qml.CNOT(wires=[i, j])
    return qml.state()


class QuantumKernelQSVC:
    """
    Quantum Support Vector Classifier leveraging a Quantum State Fidelity Kernel.
    K(x_i, x_j) = |<psi(x_i) | psi(x_j)>|^2
    """
    def __init__(
        self,
        n_qubits: int = 4,
        feature_map_type: FeatureMapType = "zz_feature_map",
        C: float = 1.0,
        random_state: int = 42
    ):
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.C = C
        self.random_state = random_state

        self.svc = SVC(kernel="precomputed", C=self.C, random_state=self.random_state)
        self.X_train_: Optional[np.ndarray] = None
        self.train_states_: Optional[torch.Tensor] = None
        self.is_fitted = False

        self.platt_a = 1.0
        self.platt_b = 0.0

    def get_quantum_state(self, x: np.ndarray) -> torch.Tensor:
        """Evaluates 2^n complex quantum statevector |psi(x)> using PyTorch."""
        x_torch = torch.as_tensor(x, dtype=torch.float64)
        if self.feature_map_type == "angle":
            return qnode_state_angle(x_torch)
        else:
            return qnode_state_zz(x_torch)

    def compute_state_vectors(self, X: np.ndarray) -> torch.Tensor:
        """Computes quantum statevectors for a batch X of shape (N, n_qubits)."""
        states = [self.get_quantum_state(x) for x in X]
        return torch.stack(states)

    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Calculates the Gram (Kernel) matrix K between dataset X1 and X2."""
        states1 = self.compute_state_vectors(X1)
        states2 = self.compute_state_vectors(X2) if X1 is not X2 else states1

        # Inner product matrix (N1, N2)
        inner_prod = torch.matmul(states1, states2.conj().T)
        fidelity = torch.abs(inner_prod) ** 2
        return fidelity.detach().cpu().numpy()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits the QSVC classifier on X_train using quantum state fidelity kernel."""
        self.X_train_ = np.array(X_train, dtype=np.float64)
        self.train_states_ = self.compute_state_vectors(self.X_train_)

        K_train = self.compute_kernel_matrix(self.X_train_, self.X_train_)
        self.svc.fit(K_train, y_train)
        self.is_fitted = True

        decision_scores = self.svc.decision_function(K_train)
        self._fit_platt_scaling(decision_scores, y_train)
        return self

    def _fit_platt_scaling(self, decision_scores: np.ndarray, y_true: np.ndarray):
        """Fits sigmoid calibration parameters (Platt scaling) for probability estimation."""
        y_targets = np.where(y_true == 1, 1.0, -1.0)
        from scipy.optimize import minimize
        def loss_fn(params):
            a, b = params
            logits = a * decision_scores + b
            p = 1.0 / (1.0 + np.exp(-logits))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            y_bin = (y_targets + 1.0) / 2.0
            return -np.sum(y_bin * np.log(p) + (1.0 - y_bin) * np.log(1.0 - p))

        res = minimize(loss_fn, [1.0, 0.0], method="BFGS")
        if res.success:
            self.platt_a, self.platt_b = res.x
        else:
            self.platt_a, self.platt_b = 1.0, 0.0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        K_test = self.compute_kernel_matrix(X, self.X_train_)
        return self.svc.decision_function(K_test)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        K_test = self.compute_kernel_matrix(X, self.X_train_)
        return self.svc.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dec = self.decision_function(X)
        logits = self.platt_a * dec + self.platt_b
        p1 = 1.0 / (1.0 + np.exp(-logits))
        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X_test)
        probas = self.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, probas)
        except Exception:
            auc = 0.5
        cm = confusion_matrix(y_test, preds).tolist()
        return {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm
        }
