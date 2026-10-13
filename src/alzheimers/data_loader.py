"""
src/alzheimers/data_loader.py
==============================================================================
Data Loader for Alzheimer's Disease (NeuroScan-AD + OASIS Clinical Cohort)
==============================================================================
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image

_SRC_DIR = Path(__file__).resolve().parent
_ROOT = _SRC_DIR.parent.parent
ALZHEIMER_LARGE_DIR = _ROOT / "data" / "Alzheimer_s Dataset"
DEFAULT_MRI_DIR = (ALZHEIMER_LARGE_DIR / "train") if (ALZHEIMER_LARGE_DIR / "train").exists() else (_ROOT / "data" / "alzheimers" / "mri_images")
DEFAULT_OASIS_CSV = _ROOT / "data" / "alzheimers" / "oasis_cross-sectional.csv"

STAGE_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
STAGE_MAP = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3,
}
STAGE_DESCRIPTIONS = {
    "NonDemented": "Normal cognitive health. No sonographic/MRI indicators of cortical atrophy or hippocampal loss.",
    "VeryMildDemented": "Very early stage impairment (CDR 0.5). Minimal hippocampal volume loss and mild memory recall lapses.",
    "MildDemented": "Mild Alzheimer's disease (CDR 1.0). Evident ventricular enlargement and moderate cerebral cortex thinning.",
    "ModerateDemented": "Moderate Alzheimer's disease (CDR 2.0). Pronounced neurodegeneration, widespread cortical atrophy and ventricular dilation.",
}


def load_alzheimers_mri(
    base_path: Optional[Path] = None,
    image_size: Tuple[int, int] = (64, 64),
    binary: bool = True,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Loads brain MRI images across the 4 cognitive categories.
    
    Returns
    -------
    X : np.ndarray shape (N, image_size[0] * image_size[1])
    y : np.ndarray shape (N,) — 0 for NonDemented, 1 for Demented (if binary)
    filenames : List of image filenames
    stages : List of original 4-stage indices (0 to 3)
    """
    path = Path(base_path) if base_path else DEFAULT_MRI_DIR
    if not path.exists():
        raise FileNotFoundError(f"Alzheimer's MRI directory not found: {path}")

    X = []
    y = []
    filenames = []
    stages = []

    for stage_name, stage_idx in STAGE_MAP.items():
        folder = path / stage_name
        if not folder.exists():
            continue
        all_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if max_samples_per_class is not None and max_samples_per_class > 0:
            all_files = all_files[:max_samples_per_class]

        for file in all_files:
            img_path = folder / file
            try:
                img = Image.open(img_path).convert('L').resize(image_size)
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                filenames.append(file)
                stages.append(stage_idx)
                label = 0 if stage_idx == 0 else 1 if binary else stage_idx
                y.append(label)
            except Exception as e:
                print(f"[Loader] Error loading {img_path}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), filenames, stages


def load_oasis_clinical_data(
    csv_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Loads and cleans the OASIS cross-sectional clinical cohort (436 patients).
    
    Features extracted:
      - Age (years)
      - Educ (Years of Education)
      - SES (Socioeconomic Status)
      - MMSE (Mini-Mental State Examination, 0–30)
      - eTIV (Estimated Total Intracranial Volume, mm³)
      - nWBV (Normalized Whole Brain Volume, %)
      
    Target:
      - CDR (Clinical Dementia Rating): 0 = Normal, 1 = Dementia (CDR >= 0.5)
    """
    path = Path(csv_path) if csv_path else DEFAULT_OASIS_CSV
    if not path.exists():
        raise FileNotFoundError(f"OASIS clinical CSV not found: {path}")

    df = pd.read_csv(path)
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    feature_cols = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV']
    
    # Impute missing values with median
    for col in feature_cols:
        if col in df.columns:
            median_val = df[col].median(skipna=True)
            df[col] = df[col].fillna(median_val)

    # Impute CDR missing values (usually missing means Non-Demented or control)
    if 'CDR' in df.columns:
        df['CDR'] = df['CDR'].fillna(0.0)
        y = (df['CDR'] >= 0.5).astype(int).values
    else:
        y = np.zeros(len(df), dtype=int)

    X = df[feature_cols].values.astype(np.float32)

    return X, y, feature_cols, df
