"""
quantum_model.py - Quantum Support Vector Classifier (QSVC) using PennyLane and PyTorch

Implements:
1. PennyLane Quantum Circuit with AngleEmbedding and ZZFeatureMap (entanglement).
2. Quantum State Fidelity Kernel: K(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2.
3. PyTorch tensor accelerated quantum state operations.
4. Scikit-Learn Precomputed Kernel Support Vector Classifier (QSVC) with Platt probability calibration.
"""

import numpy as np
import torch
from typing import Literal, Optional, Tuple, Dict, Any
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

FeatureMapType = Literal["angle", "zz_feature_map"]

# Global PennyLane simulator device on 4 qubits
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
    
    The kernel evaluates transition probabilities in quantum Hilbert space:
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

        # Classical precomputed SVM classifier
        self.svc = SVC(kernel="precomputed", C=self.C, random_state=self.random_state)
        self.X_train_: Optional[np.ndarray] = None
        self.train_states_: Optional[torch.Tensor] = None
        self.is_fitted = False

        # Platt scaling parameters for calibrated probability estimation
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
        """
        Computes quantum statevectors for a batch X of shape (N, n_qubits).
        Returns complex torch tensor of shape (N, 2^n_qubits).
        """
        states = [self.get_quantum_state(x) for x in X]
        return torch.stack(states)

    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculates the Gram (Kernel) matrix K between dataset X1 and X2:
            K_{i,j} = | <psi(X1_i) | psi(X2_j)> |^2
        """
        states1 = self.compute_state_vectors(X1)
        states2 = self.compute_state_vectors(X2)

        # PyTorch inner product: <psi_1 | psi_2> = states1 @ states2.conj().T
        inner_prod = torch.matmul(states1, states2.conj().T)
        kernel_matrix = torch.abs(inner_prod) ** 2
        K_np = kernel_matrix.detach().cpu().numpy()

        if X1 is X2 or (X1.shape == X2.shape and np.allclose(X1, X2)):
            np.fill_diagonal(K_np, 1.0)
        return np.clip(K_np, 0.0, 1.0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Trains QSVC:
        1. Computes quantum Gram matrix K_train (N_train x N_train).
        2. Fits the support vector classifier.
        3. Calibrates Platt sigmoid for probability outputs.
        """
        self.X_train_ = np.copy(X_train)
        self.train_states_ = self.compute_state_vectors(self.X_train_)
        K_train = self.compute_kernel_matrix(self.X_train_, self.X_train_)
        self.svc.fit(K_train, y_train)

        # Fit Platt calibration on decision values
        dec = self.svc.decision_function(K_train)
        # Numerical Platt scaling fit
        from scipy.optimize import minimize
        def log_loss(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * dec + b)))
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            return -np.sum(y_train * np.log(p) + (1.0 - y_train) * np.log(1.0 - p))

        res = minimize(log_loss, [1.5, 0.0], method="Nelder-Mead")
        self.platt_a, self.platt_b = float(res.x[0]), float(res.x[1])

        self.is_fitted = True
        return K_train

    def _kernel_against_train(self, X: np.ndarray) -> np.ndarray:
        """Computes K(X, X_train) using cached training statevectors."""
        states_x = self.compute_state_vectors(X)
        inner = torch.matmul(states_x, self.train_states_.conj().T)
        K = torch.abs(inner) ** 2
        return np.clip(K.detach().cpu().numpy(), 0.0, 1.0)

    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        """Computes distance to quantum hyperplane."""
        if not self.is_fitted:
            raise RuntimeError("QSVC must be trained before predicting.")
        K_test = self._kernel_against_train(X_test)
        return self.svc.decision_function(K_test)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predicts binary CHD class (0 = Low Risk, 1 = High Risk)."""
        if not self.is_fitted:
            raise RuntimeError("QSVC must be trained before predicting.")
        K_test = self._kernel_against_train(X_test)
        return self.svc.predict(K_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predicts calibrated probabilities [P(Low Risk), P(High Risk)]."""
        dec = self.decision_function(X_test)
        p1 = 1.0 / (1.0 + np.exp(-(self.platt_a * dec + self.platt_b)))
        p1 = np.clip(p1, 0.001, 0.999)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates classifier performance and returns metrics."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]

        return {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
            "roc_curve": roc_points,
            "n_support_vectors": int(len(self.svc.support_)),
            "support_vector_indices": self.svc.support_.tolist()
        }

    def get_circuit_spec(self, sample_angles: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Returns visual representation of quantum gates."""
        if sample_angles is None:
            sample_angles = np.array([0.5, -0.8, 1.2, -0.3])

        gates = []
        if self.feature_map_type == "angle":
            for i in range(self.n_qubits):
                gates.append({
                    "gate": "RY",
                    "qubit": i,
                    "target": None,
                    "param": round(float(sample_angles[i]), 3),
                    "description": f"Feature {i} rotation angle"
                })
        else:
            for i in range(self.n_qubits):
                gates.append({"gate": "H", "qubit": i, "target": None, "param": None, "description": "Equal Superposition"})
            for i in range(self.n_qubits):
                gates.append({
                    "gate": "RZ",
                    "qubit": i,
                    "target": None,
                    "param": round(float(2.0 * sample_angles[i]), 3),
                    "description": f"Feature {i} phase"
                })
            for i in range(self.n_qubits):
                j = (i + 1) % self.n_qubits
                phi = 2.0 * (np.pi - sample_angles[i]) * (np.pi - sample_angles[j])
                gates.append({"gate": "CNOT", "qubit": i, "target": j, "param": None, "description": f"Entangle Q{i}-Q{j}"})
                gates.append({"gate": "RZ", "qubit": j, "target": None, "param": round(float(phi), 3), "description": f"Entanglement phase Q{i}*Q{j}"})
                gates.append({"gate": "CNOT", "qubit": i, "target": j, "param": None, "description": f"Unentangle Q{i}-Q{j}"})

        return {
            "n_qubits": self.n_qubits,
            "feature_map": self.feature_map_type,
            "gates": gates
        }
