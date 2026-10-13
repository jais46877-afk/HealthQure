"""
src/database/storage.py
==============================================================================
Persistent SQLite Storage Engine for Clinical Diagnostic Runs
==============================================================================
Stores every inference analysis across:
  - Breast Cancer (Ultrasound -> Classical / VQC / IQM Hardware)
  - Heart Disease (Cardiovascular Profile -> Havlicek QSVC / IQM Hardware)
  - Alzheimer's Disease (Brain MRI NeuroScans & OASIS Biomarkers -> QSVC / VQC / IQM)
Zero-configuration SQLite database located at: data/diagnostics.db
==============================================================================
"""

import sqlite3
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = _ROOT / "data"
DB_PATH = DB_DIR / "diagnostics.db"


def get_db_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite schema and indexes if they do not exist."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS diagnostic_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    hardware_backend TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    stage_or_risk TEXT,
                    confidence_pct REAL,
                    probabilities_json TEXT,
                    input_summary_json TEXT,
                    quantum_telemetry_json TEXT,
                    has_heatmap INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0.0,
                    notes TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_disease 
                ON diagnostic_runs(disease)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_timestamp 
                ON diagnostic_runs(timestamp DESC)
            """)
    finally:
        conn.close()


def generate_patient_id(disease: str) -> str:
    """Generates a realistic clinical patient accession ID."""
    prefix_map = {
        "breast_cancer": "PAT-BC",
        "heart_disease": "PAT-HD",
        "alzheimers_mri": "PAT-ALZ-M",
        "alzheimers_clinical": "PAT-ALZ-C",
        "alzheimers": "PAT-ALZ"
    }
    pfx = prefix_map.get(disease, "PAT-GEN")
    num = random.randint(1000, 9999)
    return f"{pfx}-{num}"


def record_run(
    disease: str,
    modality: str,
    model_used: str,
    hardware_backend: str,
    prediction: str,
    stage_or_risk: Optional[str] = None,
    confidence_pct: Optional[float] = None,
    probabilities: Optional[Any] = None,
    input_summary: Optional[Any] = None,
    quantum_telemetry: Optional[Any] = None,
    has_heatmap: bool = False,
    latency_ms: float = 0.0,
    patient_id: Optional[str] = None,
    notes: Optional[str] = None
) -> int:
    """Inserts a diagnostic analysis run into SQLite and returns the inserted run ID."""
    init_db()
    conn = get_db_connection()
    now_iso = datetime.now(timezone.utc).isoformat()
    pid = patient_id or generate_patient_id(disease)

    prob_str = json.dumps(probabilities) if probabilities is not None else None
    inp_str = json.dumps(input_summary) if input_summary is not None else None
    telemetry_str = json.dumps(quantum_telemetry) if quantum_telemetry is not None else None

    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO diagnostic_runs (
                    timestamp, disease, patient_id, modality, model_used,
                    hardware_backend, prediction, stage_or_risk, confidence_pct,
                    probabilities_json, input_summary_json, quantum_telemetry_json,
                    has_heatmap, latency_ms, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso, disease, pid, modality, model_used,
                hardware_backend, prediction, stage_or_risk,
                round(float(confidence_pct), 2) if confidence_pct is not None else None,
                prob_str, inp_str, telemetry_str,
                1 if has_heatmap else 0,
                round(float(latency_ms), 2),
                notes
            ))
            return cursor.lastrowid
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for f in ("probabilities_json", "input_summary_json", "quantum_telemetry_json"):
        val = d.get(f)
        key = f.replace("_json", "")
        if val:
            try:
                d[key] = json.loads(val)
            except Exception:
                d[key] = None
        else:
            d[key] = None
        d.pop(f, None)
    d["has_heatmap"] = bool(d.get("has_heatmap", 0))
    return d


def get_history(
    disease: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Retrieves past diagnostic runs, optionally filtered by disease."""
    init_db()
    conn = get_db_connection()
    try:
        if disease and disease != "all":
            if disease == "alzheimers":
                cursor = conn.execute("""
                    SELECT * FROM diagnostic_runs 
                    WHERE disease IN ('alzheimers_mri', 'alzheimers_clinical', 'alzheimers')
                    ORDER BY id DESC LIMIT ? OFFSET ?
                """, (limit, offset))
            else:
                cursor = conn.execute("""
                    SELECT * FROM diagnostic_runs 
                    WHERE disease = ?
                    ORDER BY id DESC LIMIT ? OFFSET ?
                """, (disease, limit, offset))
        else:
            cursor = conn.execute("""
                SELECT * FROM diagnostic_runs 
                ORDER BY id DESC LIMIT ? OFFSET ?
            """, (limit, offset))
        return [_row_to_dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_run_by_id(run_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single diagnostic run by ID."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM diagnostic_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_run(run_id: int) -> bool:
    """Deletes a single run from the database."""
    init_db()
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM diagnostic_runs WHERE id = ?", (run_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_history(disease: Optional[str] = None) -> int:
    """Clears run history, optionally for a specific disease."""
    init_db()
    conn = get_db_connection()
    try:
        with conn:
            if disease and disease != "all":
                cursor = conn.execute("DELETE FROM diagnostic_runs WHERE disease = ?", (disease,))
            else:
                cursor = conn.execute("DELETE FROM diagnostic_runs")
            return cursor.rowcount
    finally:
        conn.close()


def get_summary_stats() -> Dict[str, Any]:
    """Computes aggregated analytics across all past diagnostic runs."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM diagnostic_runs")
        total_runs = cursor.fetchone()[0]

        cursor = conn.execute("""
            SELECT disease, COUNT(*) as count 
            FROM diagnostic_runs 
            GROUP BY disease
        """)
        by_disease = {r["disease"]: r["count"] for r in cursor.fetchall()}

        cursor = conn.execute("""
            SELECT hardware_backend, COUNT(*) as count 
            FROM diagnostic_runs 
            GROUP BY hardware_backend
        """)
        by_hardware = {r["hardware_backend"]: r["count"] for r in cursor.fetchall()}

        cursor = conn.execute("""
            SELECT prediction, COUNT(*) as count 
            FROM diagnostic_runs 
            GROUP BY prediction
        """)
        by_prediction = {r["prediction"]: r["count"] for r in cursor.fetchall()}

        return {
            "total_diagnostic_runs": total_runs,
            "runs_by_disease": by_disease,
            "runs_by_hardware": by_hardware,
            "runs_by_prediction": by_prediction,
            "database_path": str(DB_PATH),
        }
    finally:
        conn.close()
