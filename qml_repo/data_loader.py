"""
data_loader.py - Data Pipeline and Preprocessing for CHD Prediction

Provides data loading, clinical CHD dataset generation (aligned with Framingham & Cleveland Heart Study benchmarks),
scaling via StandardScaler, and PCA/feature alignment for PennyLane Quantum Circuits (4 Qubits).
Incorporates:
- age: Patient age in years (25-85)
- sex: Biological sex (1 = Male, 0 = Female)
- cholesterol: Serum cholesterol in mg/dL (<200 Desirable, 200-239 Borderline, >=240 High)
- resting_bp: Resting blood pressure in mm Hg (<120 Normal, 120-129 Elevated, >=130 Hypertensive)
- exercise_angina: Exercise-induced angina (0 = No, 1 = Yes)
- st_depression: ST segment depression induced by exercise (Oldpeak in mm)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 6 Clinical feature definitions
FEATURE_NAMES = [
    "age",                 # Patient age in years
    "sex",                 # Biological sex: 1 = Male, 0 = Female
    "cholesterol",         # Cholesterol in mg/dL (Normal < 200, Borderline 200-239, High >= 240)
    "resting_bp",          # Resting Blood Pressure in mm Hg (Normal < 120, Elevated 120-129, Hypertensive >= 130)
    "exercise_angina",     # Exercise Induced Angina (0 = No, 1 = Yes)
    "st_depression"        # ST-Segment Depression / Oldpeak (0.0 to 6.0 mm, indicative of myocardial ischemia)
]

TARGET_NAME = "chd_risk"   # 0 = Low/Moderate Risk (CAD Negative), 1 = High Risk (CAD Positive)


def generate_chd_dataset(
    n_samples: int = 180,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a balanced, medically grounded dataset mimicking the Cleveland Heart Disease
    clinical study distributions, incorporating Age and Sex alongside the 4 cardiovascular markers.
    """
    rng = np.random.RandomState(random_state)

    # 1. Demographic distributions
    age = rng.normal(loc=54.5, scale=9.5, size=n_samples)
    age = np.clip(age, 25.0, 80.0)

    # Biological sex: 1 = Male (approx 65%), 0 = Female (approx 35%)
    sex = (rng.uniform(0.0, 1.0, size=n_samples) < 0.65).astype(int)

    # 2. Cardiovascular measurements
    chol = rng.normal(loc=242.0, scale=46.0, size=n_samples)
    chol = np.clip(chol, 130.0, 420.0)

    bp = rng.normal(loc=131.0, scale=16.0, size=n_samples)
    bp = np.clip(bp, 95.0, 195.0)

    # Underlying cardiovascular stress score
    latent_stress = (
        0.035 * (age - 50.0) +
        0.45 * (sex - 0.5) +
        0.018 * (chol - 215.0) +
        0.024 * (bp - 124.0) +
        rng.normal(0.0, 0.4, size=n_samples)
    )

    # Exercise Induced Angina (higher probability with higher stress)
    p_angina = 1.0 / (1.0 + np.exp(-(latent_stress - 0.25)))
    angina = (rng.uniform(0.0, 1.0, size=n_samples) < p_angina).astype(int)

    # ST Depression (Oldpeak, mm)
    base_st = rng.exponential(scale=0.65, size=n_samples)
    st_depression = base_st + 1.25 * angina + 0.3 * np.maximum(0, latent_stress)
    st_depression = np.clip(st_depression, 0.0, 5.5)

    # Multi-factorial clinical risk index based on Cleveland Clinic CAD criteria
    clinical_risk = (
        0.035 * (age - 50.0) +
        0.55 * (sex - 0.5) +
        0.016 * (chol - 210.0) +
        0.022 * (bp - 125.0) +
        1.75 * angina +
        1.35 * st_depression +
        0.75 * (angina * (st_depression >= 1.2))
    )

    # Median threshold to ensure balanced training split for QSVC
    median_thresh = np.median(clinical_risk)
    y = (clinical_risk >= median_thresh).astype(int)

    # Minimal real-world noise (3%)
    flip_mask = rng.uniform(0.0, 1.0, size=n_samples) < 0.03
    y[flip_mask] = 1 - y[flip_mask]

    df = pd.DataFrame({
        "age": np.round(age, 1),
        "sex": sex,
        "cholesterol": np.round(chol, 1),
        "resting_bp": np.round(bp, 1),
        "exercise_angina": angina,
        "st_depression": np.round(st_depression, 2),
        "chd_risk": y
    })
    return df


class CHDDataPipeline:
    """
    Standardizes clinical metrics and maps them to the quantum rotation interval [0, pi]
    for PennyLane statevector embedding.
    """
    def __init__(self, n_qubits: int = 4, use_pca: bool = True, random_state: int = 42):
        self.n_qubits = n_qubits
        self.use_pca = use_pca
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_qubits) if use_pca else None
        self.is_fitted = False

    def _to_quantum_angles(self, projected: np.ndarray) -> np.ndarray:
        """
        Maps unbounded orthogonal components into [0.1, pi - 0.1]
        using a smooth arctangent sigmoid transform.
        Prevents phase boundary degeneracies while preserving monotonicity.
        """
        norm = (np.arctan(projected) / np.pi) + 0.5  # maps to (0, 1)
        angles = 0.1 + norm * (np.pi - 0.2)         # maps to [0.1, pi - 0.1]
        return np.clip(angles, 0.05, np.pi - 0.05)

    def fit_transform_train_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = df[FEATURE_NAMES].values
        y = df[TARGET_NAME].values

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # 1. Standardize numeric features
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)

        # 2. PCA orthogonalization / dimensionality matching to n_qubits
        if self.use_pca and self.pca is not None:
            X_train_proj = self.pca.fit_transform(X_train_scaled)
            X_test_proj = self.pca.transform(X_test_scaled)
        else:
            X_train_proj = X_train_scaled[:, :self.n_qubits]
            X_test_proj = X_test_scaled[:, :self.n_qubits]

        # 3. Map to quantum rotation angles [0, pi]
        X_train_q = self._to_quantum_angles(X_train_proj)
        X_test_q = self._to_quantum_angles(X_test_proj)

        self.is_fitted = True
        return X_train_q, X_test_q, y_train, y_test

    def transform_single_patient(self, patient_dict: Dict[str, Any]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming patient input.")

        age_val = float(patient_dict.get("age", 54.0))
        # Handle string ("male"/"female") or int (1/0)
        sex_raw = patient_dict.get("sex", 1)
        if isinstance(sex_raw, str):
            sex_val = 1.0 if sex_raw.lower().startswith("m") else 0.0
        else:
            sex_val = float(sex_raw)

        chol_val = float(patient_dict.get("cholesterol", 200.0))
        bp_val = float(patient_dict.get("resting_bp", 120.0))
        ang_val = float(patient_dict.get("exercise_angina", 0))
        st_val = float(patient_dict.get("st_depression", 0.0))

        raw_vec = np.array([[
            age_val,
            sex_val,
            chol_val,
            bp_val,
            ang_val,
            st_val
        ]])

        scaled = self.scaler.transform(raw_vec)
        if self.use_pca and self.pca is not None:
            proj = self.pca.transform(scaled)
        else:
            proj = scaled[:, :self.n_qubits]

        quantum_angles = self._to_quantum_angles(proj)
        return quantum_angles[0]

    def export_state(self) -> Dict[str, Any]:
        state = {
            "n_qubits": self.n_qubits,
            "use_pca": self.use_pca,
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else [],
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else [],
            "feature_names": FEATURE_NAMES
        }
        if self.pca is not None and hasattr(self.pca, "components_"):
            state["pca_components"] = self.pca.components_.tolist()
            state["pca_mean"] = self.pca.mean_.tolist()
            state["pca_explained_variance_ratio"] = self.pca.explained_variance_ratio_.tolist()
        return state

    def load_state(self, state: Dict[str, Any]):
        self.n_qubits = state["n_qubits"]
        self.use_pca = state["use_pca"]
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(state["scaler_mean"])
        self.scaler.scale_ = np.array(state["scaler_scale"])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_samples_seen_ = 180

        if self.use_pca and "pca_components" in state:
            self.pca = PCA(n_components=self.n_qubits)
            self.pca.components_ = np.array(state["pca_components"])
            self.pca.mean_ = np.array(state["pca_mean"])
            self.pca.explained_variance_ratio_ = np.array(state["pca_explained_variance_ratio"])
        self.is_fitted = True
