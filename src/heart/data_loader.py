"""
src/heart/data_loader.py
==============================================================================
Data Pipeline and Preprocessing for Coronary Heart Disease (CHD) Prediction
==============================================================================
Feature schema:
  1. age             (years)
  2. sex             (1 = Male, 0 = Female)
  3. cholesterol     (mg/dL)
  4. resting_bp      (mm Hg)
  5. exercise_angina (0 = No, 1 = Yes)
  6. st_depression   (Oldpeak in mm)
==============================================================================
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURE_NAMES = [
    "age",
    "sex",
    "cholesterol",
    "resting_bp",
    "exercise_angina",
    "st_depression"
]

TARGET_NAME = "chd_risk"


def generate_chd_dataset(
    n_samples: int = 200,
    random_state: int = 42
) -> pd.DataFrame:
    """Generates a balanced clinical CHD dataset based on Cleveland Study distributions."""
    rng = np.random.RandomState(random_state)

    age = rng.normal(loc=54.5, scale=9.5, size=n_samples)
    age = np.clip(age, 25.0, 80.0)

    sex = (rng.uniform(0.0, 1.0, size=n_samples) < 0.65).astype(int)

    chol = rng.normal(loc=242.0, scale=46.0, size=n_samples)
    chol = np.clip(chol, 130.0, 420.0)

    bp = rng.normal(loc=131.0, scale=16.0, size=n_samples)
    bp = np.clip(bp, 95.0, 195.0)

    latent_stress = (
        0.035 * (age - 50.0) +
        0.45 * (sex - 0.5) +
        0.018 * (chol - 215.0) +
        0.024 * (bp - 124.0) +
        rng.normal(0.0, 0.4, size=n_samples)
    )

    p_angina = 1.0 / (1.0 + np.exp(-(latent_stress - 0.25)))
    angina = (rng.uniform(0.0, 1.0, size=n_samples) < p_angina).astype(int)

    base_st = rng.exponential(scale=0.65, size=n_samples)
    st_depression = base_st + 1.25 * angina + 0.3 * np.maximum(0, latent_stress)
    st_depression = np.clip(st_depression, 0.0, 5.5)

    clinical_risk = (
        0.035 * (age - 50.0) +
        0.55 * (sex - 0.5) +
        0.016 * (chol - 210.0) +
        0.022 * (bp - 125.0) +
        1.75 * angina +
        1.35 * st_depression +
        0.75 * (angina * (st_depression >= 1.2))
    )

    median_thresh = np.median(clinical_risk)
    y = (clinical_risk >= median_thresh).astype(int)

    flip_mask = rng.uniform(0.0, 1.0, size=n_samples) < 0.03
    y[flip_mask] = 1 - y[flip_mask]

    return pd.DataFrame({
        "age": np.round(age, 1),
        "sex": sex,
        "cholesterol": np.round(chol, 1),
        "resting_bp": np.round(bp, 1),
        "exercise_angina": angina,
        "st_depression": np.round(st_depression, 2),
        TARGET_NAME: y
    })


class CHDDataPipeline:
    """Preprocesses 6 clinical attributes and projects to 4 quantum qubit angles."""

    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = X[FEATURE_NAMES].values
        X_scaled = self.scaler.fit_transform(X_mat)
        X_pca = self.pca.fit_transform(X_scaled)

        # Scale PCA features into angles (-pi, pi) for quantum gate embedding
        X_angles = np.pi * (X_pca / (np.max(np.abs(X_pca), axis=0) + 1e-8))
        self.is_fitted = True
        return X_angles

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit_transform first.")
        X_mat = X[FEATURE_NAMES].values
        X_scaled = self.scaler.transform(X_mat)
        X_pca = self.pca.transform(X_scaled)
        X_angles = np.pi * (X_pca / (np.max(np.abs(X_pca), axis=0) + 1e-8))
        return X_angles
