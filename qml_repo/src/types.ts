export interface ClinicalInput {
  age: number;
  sex: number; // 1 = Male, 0 = Female
  cholesterol: number;
  resting_bp: number;
  exercise_angina: number;
  st_depression: number;
  feature_map: "zz_feature_map" | "angle";
}

export interface SupportVectorFidelity {
  sv_index: number;
  fidelity: number;
  sv_class: "High Risk" | "Low Risk";
}

export interface QubitMapping {
  qubit: number;
  feature: string;
  angle_rad: number;
}

export interface QuantumGate {
  gate: string;
  qubit: number;
  target: number | null;
  param: number | null;
  description: string;
}

export interface CircuitSpec {
  n_qubits: number;
  feature_map: string;
  gates: QuantumGate[];
}

export interface PredictionResult {
  prediction: number;
  cad_diagnosis?: "CAD Positive" | "CAD Negative" | string;
  cad_status?: "POSITIVE" | "NEGATIVE";
  prediction_label: string;
  risk_level: "High" | "Moderate" | "Low";
  chd_probability: number;
  low_risk_probability: number;
  decision_margin: number;
  clinical_interpretation: string;
  patient_metrics?: {
    age: number;
    sex: string;
    sex_code: number;
    cholesterol: number;
    resting_bp: number;
    exercise_angina: number;
    st_depression: number;
  };
  quantum_encoding: {
    n_qubits: number;
    feature_map: string;
    qubit_angles_rad: number[];
    qubit_angles_deg: number[];
    qubit_mapping: QubitMapping[];
  };
  support_vector_fidelities: SupportVectorFidelity[];
  circuit_spec: CircuitSpec;
}

export interface BatchPatientRecord {
  id: string;
  name?: string;
  age?: number;
  sex?: string | number;
  cholesterol: number;
  resting_bp: number;
  exercise_angina: number;
  st_depression: number;
  prediction?: number;
  cad_diagnosis?: "CAD Positive" | "CAD Negative" | string;
  cad_status?: "POSITIVE" | "NEGATIVE";
  prediction_label?: string;
  risk_level?: "High" | "Moderate" | "Low";
  chd_probability?: number;
  low_risk_probability?: number;
  decision_margin?: number;
}

export interface BatchPredictionResponse {
  count: number;
  positive_count: number;
  negative_count: number;
  results: BatchPatientRecord[];
}

export interface RocCurvePoint {
  fpr: number;
  tpr: number;
}

export interface ModelMetrics {
  accuracy: number;
  f1_score: number;
  roc_auc: number;
  confusion_matrix: number[][];
  roc_curve: RocCurvePoint[];
  n_support_vectors: number;
  support_vector_indices: number[];
}

export interface ModelStatus {
  status: string;
  simulator: string;
  n_qubits: number;
  feature_map_type: string;
  n_train_samples: number;
  n_support_vectors: number;
  metrics: ModelMetrics;
}

export interface DatasetResponse {
  total_records: number;
  positive_count: number;
  negative_count: number;
  sample_records: Array<{
    age?: number;
    sex?: number;
    cholesterol: number;
    resting_bp: number;
    exercise_angina: number;
    st_depression: number;
    chd_risk: number;
  }>;
}
