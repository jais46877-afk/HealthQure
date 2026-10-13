import React, { useState, useEffect } from "react";
import {
  ClinicalInput,
  PredictionResult,
  ModelStatus,
  DatasetResponse,
  ModelMetrics
} from "./types";
import { PatientForm } from "./components/PatientForm";
import { ResultCard } from "./components/ResultCard";
import { MetricsView } from "./components/MetricsView";
import { DatasetView } from "./components/DatasetView";
import { SourceArchitecture } from "./components/SourceArchitecture";
import { ClinicalDataUploader } from "./components/ClinicalDataUploader";
import {
  Activity,
  Cpu,
  BarChart3,
  Database,
  Code2,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  FileSpreadsheet
} from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState<"predict" | "batch" | "metrics" | "dataset" | "code">("predict");

  // Form input state
  const [patientInput, setPatientInput] = useState<ClinicalInput>({
    age: 58,
    sex: 1,
    cholesterol: 245,
    resting_bp: 135,
    exercise_angina: 1,
    st_depression: 1.8,
    feature_map: "zz_feature_map"
  });

  // Quantum API states
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [gramMatrix, setGramMatrix] = useState<number[][]>([]);
  const [dataset, setDataset] = useState<DatasetResponse | undefined>(undefined);

  const [predicting, setPredicting] = useState<boolean>(false);
  const [retraining, setRetraining] = useState<boolean>(false);
  const [notification, setNotification] = useState<{ type: "success" | "error"; msg: string } | null>(null);

  // Initialize data
  useEffect(() => {
    fetchStatus();
    fetchDataset();
    // Run initial prediction for immediate view
    executePrediction(patientInput);
  }, []);

  const showNotification = (type: "success" | "error", msg: string) => {
    setNotification({ type, msg });
    setTimeout(() => setNotification(null), 4000);
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        setModelStatus(data);
      }
      const mRes = await fetch("/api/metrics");
      if (mRes.ok) {
        const mData = await mRes.json();
        if (mData.sample_gram_matrix) {
          setGramMatrix(mData.sample_gram_matrix);
        }
      }
    } catch (err) {
      console.error("Failed to load status:", err);
    }
  };

  const fetchDataset = async () => {
    try {
      const res = await fetch("/api/dataset");
      if (res.ok) {
        const data = await res.json();
        setDataset(data);
      }
    } catch (err) {
      console.error("Failed to fetch dataset:", err);
    }
  };

  const executePrediction = async (input: ClinicalInput) => {
    setPredicting(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input)
      });
      if (res.ok) {
        const data = await res.json();
        setPredictionResult(data);
      } else {
        const err = await res.json();
        showNotification("error", err.details || "Inference failed");
      }
    } catch (err: any) {
      showNotification("error", "Error connecting to Quantum inference backend");
    } finally {
      setPredicting(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executePrediction(patientInput);
  };

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          n_samples: 180,
          feature_map_type: patientInput.feature_map
        })
      });
      if (res.ok) {
        const data = await res.json();
        showNotification("success", `Quantum Model Retrained! Test ROC-AUC: ${(data.metrics.roc_auc || 0).toFixed(3)}`);
        fetchStatus();
      } else {
        showNotification("error", "Retraining failed");
      }
    } catch (err) {
      showNotification("error", "Failed to trigger quantum training");
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 flex flex-col font-sans">
      {/* Toast Notification */}
      {notification && (
        <div
          className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-lg shadow-lg border text-xs font-medium flex items-center gap-2 animate-in fade-in slide-in-from-top-2 ${
            notification.type === "success"
              ? "bg-emerald-50 text-emerald-900 border-emerald-300"
              : "bg-rose-50 text-rose-900 border-rose-300"
          }`}
        >
          {notification.type === "success" ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          ) : (
            <AlertCircle className="w-4 h-4 text-rose-600" />
          )}
          <span>{notification.msg}</span>
        </div>
      )}

      {/* Main Top Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Brand / Logo */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-sm">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-base font-bold text-slate-900 tracking-tight">
                    Quantum CHD Risk Predictor
                  </h1>
                  <span className="hidden sm:inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 uppercase">
                    PennyLane + PyTorch
                  </span>
                </div>
                <p className="text-xs text-slate-500 hidden sm:block">
                  Coronary Heart Disease Classifier via 4-Qubit State Fidelity Quantum Kernel
                </p>
              </div>
            </div>

            {/* Status & Action */}
            <div className="flex items-center gap-3">
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-slate-50 rounded-lg border border-slate-200 text-xs">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="text-slate-600 font-mono">default.qubit (4 Wires)</span>
                {modelStatus?.metrics && (
                  <span className="text-indigo-600 font-semibold border-l border-slate-200 pl-2">
                    AUC: {modelStatus.metrics.roc_auc.toFixed(3)}
                  </span>
                )}
              </div>

              <button
                id="header-retrain-btn"
                onClick={handleRetrain}
                disabled={retraining}
                title="Retrain PennyLane QSVC model on synthetic clinical dataset"
                className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 active:bg-slate-950 text-white text-xs font-semibold rounded-lg shadow-sm transition-all cursor-pointer disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${retraining ? "animate-spin" : ""}`} />
                <span>{retraining ? "Training..." : "Retrain QSVC"}</span>
              </button>
            </div>
          </div>

          {/* Navigation Bar */}
          <div className="flex items-center gap-1 border-t border-slate-100 overflow-x-auto py-1">
            <button
              id="tab-predict"
              onClick={() => setActiveTab("predict")}
              className={`flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-lg transition-colors cursor-pointer shrink-0 ${
                activeTab === "predict"
                  ? "bg-indigo-50 text-indigo-700 font-bold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Enter Clinical Data & Predict CAD</span>
            </button>

            <button
              id="tab-batch"
              onClick={() => setActiveTab("batch")}
              className={`flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-lg transition-colors cursor-pointer shrink-0 ${
                activeTab === "batch"
                  ? "bg-indigo-50 text-indigo-700 font-bold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <FileSpreadsheet className="w-4 h-4" />
              <span>Batch Cohort Classification (CSV/JSON)</span>
            </button>

            <button
              id="tab-metrics"
              onClick={() => setActiveTab("metrics")}
              className={`flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-lg transition-colors cursor-pointer shrink-0 ${
                activeTab === "metrics"
                  ? "bg-indigo-50 text-indigo-700 font-bold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Model Performance & ROC</span>
            </button>

            <button
              id="tab-dataset"
              onClick={() => setActiveTab("dataset")}
              className={`flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-lg transition-colors cursor-pointer shrink-0 ${
                activeTab === "dataset"
                  ? "bg-indigo-50 text-indigo-700 font-bold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <Database className="w-4 h-4" />
              <span>Clinical Cohort Data</span>
            </button>

            <button
              id="tab-code"
              onClick={() => setActiveTab("code")}
              className={`flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-lg transition-colors cursor-pointer shrink-0 ${
                activeTab === "code"
                  ? "bg-indigo-50 text-indigo-700 font-bold"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
              }`}
            >
              <Cpu className="w-4 h-4" />
              <span>Quantum Circuits & Heatmaps</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === "predict" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Left Column: Patient Clinical Input Form */}
              <div className="lg:col-span-5 space-y-6">
                <PatientForm
                  input={patientInput}
                  onChange={(updated) => setPatientInput((prev) => ({ ...prev, ...updated }))}
                  onSubmit={handleFormSubmit}
                  loading={predicting}
                />

                {/* Quantum Kernel Overview Card */}
                <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm text-xs space-y-2.5">
                  <div className="font-semibold text-slate-800 flex items-center gap-1.5">
                    <Cpu className="w-4 h-4 text-indigo-600" />
                    <span>Quantum Kernel Support Vector Machine</span>
                  </div>
                  <p className="text-slate-600 leading-relaxed">
                    Maps patient cardiovascular metrics $x$ into quantum Hilbert space $|\psi(x)\rangle$.
                    The state fidelity kernel computes inner product $K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$
                    evaluated via PennyLane statevectors in PyTorch.
                  </p>
                  <div className="font-mono text-[11px] bg-slate-50 p-2.5 rounded border border-slate-200 text-slate-700">
                    f(x) = ∑ α_i y_i |⟨ψ(x) | ψ(x_i)⟩|² + b
                  </div>
                </div>
              </div>

              {/* Right Column: Prediction Result */}
              <div className="lg:col-span-7 space-y-6">
                {predictionResult ? (
                  <div className="space-y-4">
                    <ResultCard result={predictionResult} patientInput={patientInput} />

                    {/* Direct Quick Link to Circuit & Heatmap Viewer */}
                    <div className="p-4 bg-indigo-50/70 border border-indigo-200 rounded-xl flex items-center justify-between gap-3 text-xs">
                      <div>
                        <span className="font-bold text-indigo-900 block">
                          Explore Quantum Circuit & Gram Matrix Heatmaps
                        </span>
                        <span className="text-slate-600 text-[11px]">
                          View the full 4-qubit unitary gate schematic and state fidelity transition heatmaps
                        </span>
                      </div>
                      <button
                        type="button"
                        id="btn-goto-circuits-heatmaps"
                        onClick={() => setActiveTab("code")}
                        className="px-3.5 py-1.5 bg-indigo-600 text-white font-medium rounded-lg text-xs hover:bg-indigo-700 transition-colors shadow-xs shrink-0 cursor-pointer flex items-center gap-1"
                      >
                        <Cpu className="w-3.5 h-3.5" />
                        <span>Open Circuits & Heatmaps</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="bg-white p-12 rounded-xl border border-slate-200 text-center">
                    <div className="w-8 h-8 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                    <p className="text-xs text-slate-500">Evaluating PennyLane Quantum State Fidelity...</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === "batch" && (
          <div className="space-y-6">
            <ClinicalDataUploader
              activeFeatureMap={patientInput.feature_map}
              onSelectPatientForCircuit={(selected) => {
                const updated: ClinicalInput = {
                  ...patientInput,
                  cholesterol: selected.cholesterol,
                  resting_bp: selected.resting_bp,
                  exercise_angina: selected.exercise_angina,
                  st_depression: selected.st_depression,
                  age: selected.age !== undefined ? Number(selected.age) : patientInput.age,
                  sex: selected.sex !== undefined ? (selected.sex === 1 || selected.sex === "1" ? 1 : 0) : patientInput.sex
                };
                setPatientInput(updated);
                setActiveTab("code");
              }}
            />
          </div>
        )}

        {activeTab === "metrics" && (
          <div className="space-y-6">
            {modelStatus?.metrics ? (
              <MetricsView
                metrics={modelStatus.metrics}
                onRetrain={handleRetrain}
                isTraining={retraining}
              />
            ) : (
              <div className="bg-white p-8 rounded-xl border border-slate-200 text-center text-xs text-slate-500">
                Loading model metrics...
              </div>
            )}
          </div>
        )}

        {activeTab === "dataset" && (
          <div className="space-y-6">
            <DatasetView dataset={dataset} />
          </div>
        )}

        {activeTab === "code" && (
          <div className="space-y-6">
            <SourceArchitecture
              patientInput={patientInput}
              gramMatrix={gramMatrix}
              onApplyPreset={(preset) => {
                const updated = { ...patientInput, ...preset };
                setPatientInput(updated);
                executePrediction(updated);
              }}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-700">Quantum CHD Risk Predictor</span>
            <span>—</span>
            <span>PennyLane 0.45 & PyTorch 2.14 Simulator</span>
          </div>
          <div className="flex items-center gap-4 text-[11px] text-slate-400">
            <span>4 Qubits</span>
            <span>ZZ-Feature Map / Angle Embedding</span>
            <span>State Fidelity Gram Matrix</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
