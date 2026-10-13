import React, { useState, useMemo } from "react";
import { ClinicalInput } from "../types";
import {
  Cpu,
  Grid,
  Layers,
  Zap,
  Activity,
  Sliders,
  Sparkles,
  Info,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Binary,
  ArrowRight,
  TrendingUp,
  BarChart3
} from "lucide-react";

interface SourceArchitectureProps {
  patientInput: ClinicalInput;
  gramMatrix: number[][];
  onApplyPreset?: (preset: Partial<ClinicalInput>) => void;
}

// Exact PCA & Scaler weights extracted from models/qsvc_metadata.json
const SCALER_MEAN = [53.7884, 0.6429, 243.5893, 128.8357, 0.6161, 1.7833];
const SCALER_SCALE = [8.4175, 0.4792, 38.6091, 16.6391, 0.4863, 1.0525];
const FEATURE_NAMES = ["Age", "Sex", "Cholesterol", "Resting BP", "Exercise Angina", "ST Depression"];
const FEATURE_KEYS: (keyof ClinicalInput)[] = ["age", "sex", "cholesterol", "resting_bp", "exercise_angina", "st_depression"];

const PCA_COMPONENTS = [
  // PC0 -> Qubit 0 (Variance: 34.99%) - Dominated by ST Depression (0.626) & Exercise Angina (0.607)
  [0.2620, 0.0222, 0.3297, 0.2472, 0.6073, 0.6263],
  // PC1 -> Qubit 1 (Variance: 19.89%) - Dominated by Resting BP (0.727) & negative Chol (-0.468)
  [-0.1726, -0.4704, -0.4681, 0.7270, 0.0202, 0.0288],
  // PC2 -> Qubit 2 (Variance: 16.56%) - Dominated by Sex (0.761) & negative Chol (-0.558)
  [0.2613, 0.7610, -0.5584, 0.1922, 0.0532, 0.0303],
  // PC3 -> Qubit 3 (Variance: 15.40%) - Dominated by Age (0.891)
  [0.8909, -0.3690, -0.1455, -0.1134, -0.1786, -0.0652]
];

const PCA_VARIANCE = [34.99, 19.89, 16.56, 15.40];

const QUBIT_METADATA = [
  {
    qubit: 0,
    name: "Wire |q₀⟩",
    title: "Ischemic Stress Axis (PC1)",
    driver: "ST Depression (0.63) + Exercise Angina (0.61)",
    role: "Encodes reversible myocardial ischemia & stress-induced repolarization abnormalities",
    color: "indigo"
  },
  {
    qubit: 1,
    name: "Wire |q₁⟩",
    title: "Hemodynamic Load Axis (PC2)",
    driver: "Resting Blood Pressure (0.73) & Cholesterol (-0.47)",
    role: "Encodes vascular afterload, systolic pressure, and arterial perfusion gradient",
    color: "cyan"
  },
  {
    qubit: 2,
    name: "Wire |q₂⟩",
    title: "Endocrine & Lipid Axis (PC3)",
    driver: "Biological Sex (0.76) vs. Total Cholesterol (-0.56)",
    role: "Encodes baseline cardiovascular risk profile and lipid-to-sex hormone correlations",
    color: "purple"
  },
  {
    qubit: 3,
    name: "Wire |q₃⟩",
    title: "Chronological Vascular Axis (PC4)",
    driver: "Patient Age (0.89)",
    role: "Encodes biological vascular aging and lifetime cumulative arterial stiffness",
    color: "amber"
  }
];

const PRESETS: Record<string, { label: string; desc: string; values: Partial<ClinicalInput> }> = {
  current: {
    label: "Active Patient Form Data",
    desc: "Uses the live input values currently specified in the Patient Input form",
    values: {}
  },
  cad_positive: {
    label: "Severe CAD Positive Case",
    desc: "Marked ST depression (2.8mm), exercise angina, elevated BP (152 mmHg), high cholesterol (285 mg/dL)",
    values: { age: 62, sex: 1, cholesterol: 285, resting_bp: 152, exercise_angina: 1, st_depression: 2.8 }
  },
  cad_negative: {
    label: "Healthy CAD Negative Case",
    desc: "Normal ST segment (0.1mm), no angina, optimal BP (114 mmHg), normal cholesterol (175 mg/dL)",
    values: { age: 42, sex: 0, cholesterol: 175, resting_bp: 114, exercise_angina: 0, st_depression: 0.1 }
  },
  borderline: {
    label: "Borderline Ischemia Case",
    desc: "Moderate ST depression (1.4mm), positive angina, borderline BP (136 mmHg), cholesterol 238 mg/dL",
    values: { age: 55, sex: 1, cholesterol: 238, resting_bp: 136, exercise_angina: 1, st_depression: 1.4 }
  }
};

export const SourceArchitecture: React.FC<SourceArchitectureProps> = ({
  patientInput,
  gramMatrix = [],
  onApplyPreset
}) => {
  // Navigation section: both or specific view
  const [activeSection, setActiveSection] = useState<"all" | "circuits" | "heatmaps">("all");
  const [selectedFeatureMap, setSelectedFeatureMap] = useState<"zz_feature_map" | "angle">("zz_feature_map");
  const [activePresetKey, setActivePresetKey] = useState<string>("current");
  const [heatmapTab, setHeatmapTab] = useState<"gram" | "pca" | "fidelities">("gram");

  // Hover & selection states for interactive inspection
  const [selectedGate, setSelectedGate] = useState<{
    gate: string;
    qubit: number;
    target?: number | null;
    param?: number | null;
    desc: string;
    unitary: string;
  } | null>(null);

  const [selectedGramCell, setSelectedGramCell] = useState<{
    row: number;
    col: number;
    val: number;
  } | null>(null);

  const [gramFilter, setGramFilter] = useState<"all" | "diagonal" | "high" | "off_diagonal">("all");

  // Determine current active clinical input values
  const activeInput = useMemo(() => {
    if (activePresetKey === "current" || !PRESETS[activePresetKey]) {
      return patientInput;
    }
    return {
      ...patientInput,
      ...PRESETS[activePresetKey].values
    };
  }, [activePresetKey, patientInput]);

  // Compute exact 4 Qubit Angles from clinical biomarkers
  const qubitAngles = useMemo(() => {
    const rawVec = [
      activeInput.age,
      activeInput.sex,
      activeInput.cholesterol,
      activeInput.resting_bp,
      activeInput.exercise_angina,
      activeInput.st_depression
    ];

    // Standardize
    const scaled = rawVec.map((v, i) => (v - SCALER_MEAN[i]) / SCALER_SCALE[i]);

    // Project onto 4 PCA components
    const proj = [0, 1, 2, 3].map((compIdx) => {
      return scaled.reduce((sum, v, i) => sum + v * PCA_COMPONENTS[compIdx][i], 0);
    });

    // Arctan monotonic mapping to [0.05, pi - 0.05]
    return proj.map((p) => {
      const norm = Math.atan(p) / Math.PI + 0.5;
      const angle = 0.1 + norm * (Math.PI - 0.2);
      return Math.max(0.05, Math.min(Math.PI - 0.05, angle));
    });
  }, [activeInput]);

  // Calculate entangling interaction phases: Phi_ij = 2 * (pi - theta_i) * (pi - theta_j)
  const entanglingPhases = useMemo(() => {
    const phases: { i: number; j: number; phi: number }[] = [];
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      const phi = 2.0 * (Math.PI - qubitAngles[i]) * (Math.PI - qubitAngles[j]);
      phases.push({ i, j, phi });
    }
    return phases;
  }, [qubitAngles]);

  // Gram matrix statistics
  const gramStats = useMemo(() => {
    if (!gramMatrix || gramMatrix.length === 0) {
      return { count: 0, meanFidelity: 0, maxOffDiag: 0, minOffDiag: 0 };
    }
    const n = gramMatrix.length;
    let offDiagSum = 0;
    let offDiagCount = 0;
    let maxOff = 0;
    let minOff = 1;

    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        if (r !== c) {
          const v = gramMatrix[r][c];
          offDiagSum += v;
          offDiagCount++;
          if (v > maxOff) maxOff = v;
          if (v < minOff) minOff = v;
        }
      }
    }

    return {
      count: n * n,
      meanFidelity: offDiagCount > 0 ? offDiagSum / offDiagCount : 0,
      maxOffDiag: maxOff,
      minOffDiag: minOff
    };
  }, [gramMatrix]);

  const handleApplyPresetToForm = (key: string) => {
    setActivePresetKey(key);
    if (onApplyPreset && PRESETS[key]?.values) {
      onApplyPreset(PRESETS[key].values);
    }
  };

  return (
    <div id="source-architecture-view" className="space-y-8">
      {/* Top Architecture Hero Header */}
      <div className="bg-slate-900 text-white rounded-2xl p-6 sm:p-8 border border-slate-800 shadow-md relative overflow-hidden">
        <div className="absolute right-0 top-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20"></div>

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-950/80 border border-indigo-700/60 text-indigo-300 text-xs font-mono font-medium">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>PennyLane 0.45.1 • 4-Qubit QSVC Hilbert Architecture</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
              Quantum Circuit Diagrams & State Fidelity Heatmaps
            </h2>
            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
              This architecture view visualizes the complete quantum mechanical pipeline: clinical feature
              standardization, orthogonal PCA projection, 4-wire Havlíček $ZZ$-feature map quantum circuits,
              and empirical transition fidelity Gram matrix heatmaps $K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$.
            </p>
          </div>

          {/* Quick Telemetry Badges */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-2 gap-3 shrink-0">
            <div className="bg-slate-800/80 border border-slate-700/80 p-3 rounded-xl">
              <span className="text-[10px] text-slate-400 uppercase font-semibold tracking-wider block">Hilbert Dim</span>
              <span className="text-lg font-mono font-bold text-indigo-300">2⁴ = 16</span>
              <span className="text-[10px] text-slate-400 block">Statevector basis</span>
            </div>
            <div className="bg-slate-800/80 border border-slate-700/80 p-3 rounded-xl">
              <span className="text-[10px] text-slate-400 uppercase font-semibold tracking-wider block">Quantum Wires</span>
              <span className="text-lg font-mono font-bold text-emerald-400">4 Qubits</span>
              <span className="text-[10px] text-slate-400 block">All-to-all entangling</span>
            </div>
            <div className="bg-slate-800/80 border border-slate-700/80 p-3 rounded-xl">
              <span className="text-[10px] text-slate-400 uppercase font-semibold tracking-wider block">Kernel Overlap</span>
              <span className="text-lg font-mono font-bold text-amber-300">Fidelity</span>
              <span className="text-[10px] text-slate-400 block">|⟨ψ(x)|ψ(x')⟩|²</span>
            </div>
            <div className="bg-slate-800/80 border border-slate-700/80 p-3 rounded-xl">
              <span className="text-[10px] text-slate-400 uppercase font-semibold tracking-wider block">Dual Solver</span>
              <span className="text-lg font-mono font-bold text-purple-300">PyTorch C-SVC</span>
              <span className="text-[10px] text-slate-400 block">Platt probability</span>
            </div>
          </div>
        </div>

        {/* Section Switcher Tabs */}
        <div className="mt-6 pt-5 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-1.5 p-1 bg-slate-950/80 border border-slate-800 rounded-xl">
            <button
              id="arch-btn-all"
              onClick={() => setActiveSection("all")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer flex items-center gap-1.5 ${
                activeSection === "all"
                  ? "bg-indigo-600 text-white font-semibold shadow-xs"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Full Architecture</span>
            </button>

            <button
              id="arch-btn-circuits"
              onClick={() => setActiveSection("circuits")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer flex items-center gap-1.5 ${
                activeSection === "circuits"
                  ? "bg-indigo-600 text-white font-semibold shadow-xs"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>Quantum Circuit Diagrams</span>
            </button>

            <button
              id="arch-btn-heatmaps"
              onClick={() => setActiveSection("heatmaps")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer flex items-center gap-1.5 ${
                activeSection === "heatmaps"
                  ? "bg-indigo-600 text-white font-semibold shadow-xs"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Grid className="w-3.5 h-3.5" />
              <span>Quantum Heatmaps</span>
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>Simulator active: PennyLane default.qubit</span>
          </div>
        </div>
      </div>

      {/* SECTION 1: QUANTUM CIRCUIT DIAGRAMS */}
      {(activeSection === "all" || activeSection === "circuits") && (
        <section id="section-circuits" className="space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
            {/* Header & Controls Bar */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-5">
              <div>
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-indigo-50 border border-indigo-200 text-indigo-700 flex items-center justify-center font-bold text-xs">
                    4Q
                  </div>
                  <h3 className="text-base font-bold text-slate-900">
                    Interactive 4-Qubit Quantum Circuit Diagram
                  </h3>
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Parameterized unitary state preparation U_Φ(x)|0⟩^⊗4 for Coronary Heart Disease classification
                </p>
              </div>

              {/* Feature map selection */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium text-slate-600">Feature Map:</span>
                <div className="inline-flex rounded-lg bg-slate-100 p-0.5 border border-slate-200">
                  <button
                    id="fmap-btn-zz"
                    onClick={() => setSelectedFeatureMap("zz_feature_map")}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition-all cursor-pointer ${
                      selectedFeatureMap === "zz_feature_map"
                        ? "bg-white text-indigo-700 font-bold shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    ZZFeatureMap (Entangled)
                  </button>
                  <button
                    id="fmap-btn-angle"
                    onClick={() => setSelectedFeatureMap("angle")}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition-all cursor-pointer ${
                      selectedFeatureMap === "angle"
                        ? "bg-white text-indigo-700 font-bold shadow-xs"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    AngleEmbedding (Separable Ry)
                  </button>
                </div>
              </div>
            </div>

            {/* Clinical Preset Selector */}
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Select Clinical Profile to Inject into Quantum Wires:</span>
                </span>
                <span className="text-[11px] text-slate-500">
                  Dynamic rotation parameters calculated via PCA projection
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                {Object.entries(PRESETS).map(([key, item]) => {
                  const isActive = activePresetKey === key;
                  return (
                    <button
                      key={key}
                      id={`preset-btn-${key}`}
                      onClick={() => handleApplyPresetToForm(key)}
                      className={`text-left p-3 rounded-lg border text-xs transition-all cursor-pointer ${
                        isActive
                          ? "bg-white border-indigo-500 shadow-sm ring-1 ring-indigo-500"
                          : "bg-white/80 border-slate-200 hover:border-slate-300 text-slate-700"
                      }`}
                    >
                      <div className="font-bold flex items-center justify-between">
                        <span className={isActive ? "text-indigo-900" : "text-slate-900"}>{item.label}</span>
                        {isActive && <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600 shrink-0" />}
                      </div>
                      <p className="text-[10px] text-slate-500 mt-1 line-clamp-2 leading-tight">
                        {item.desc}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Circuit Diagram Canvas / Lanes */}
            <div className="bg-slate-950 text-slate-100 rounded-xl p-6 border border-slate-800 shadow-inner">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-slate-800 pb-3">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-indigo-400 font-bold">CIRCUIT SCHEMATIC</span>
                  <span className="text-xs text-slate-400">
                    Depth: {selectedFeatureMap === "zz_feature_map" ? "13 Layers (H + RZ + CNOT + RZ + CNOT)" : "3 Layers (RY + Measure)"}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
                  <span>Target: |ψ(x)⟩ ∈ ℂ¹⁶</span>
                  <span>Wires: 4</span>
                </div>
              </div>

              {/* 4 Qubit Circuit Lanes */}
              <div className="overflow-x-auto pb-4">
                <div className="min-w-[780px] space-y-6">
                  {QUBIT_METADATA.map((qMeta, qIdx) => {
                    const angle = qubitAngles[qIdx];
                    const angleDeg = ((angle * 180) / Math.PI).toFixed(1);
                    const twoTheta = (2.0 * angle).toFixed(3);
                    const entPhase = entanglingPhases.find((p) => p.i === qIdx);

                    return (
                      <div key={qIdx} className="flex items-center gap-4">
                        {/* Qubit Wire Header / Metadata */}
                        <div className="w-56 shrink-0 bg-slate-900 border border-slate-800 rounded-lg p-2.5 space-y-1 text-left">
                          <div className="flex items-center justify-between">
                            <span className="font-mono font-bold text-xs text-indigo-400">
                              {qMeta.name}
                            </span>
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                              θ = {angle.toFixed(2)} rad
                            </span>
                          </div>
                          <div className="text-[11px] font-semibold text-slate-200 truncate" title={qMeta.title}>
                            {qMeta.title}
                          </div>
                          <div className="text-[9px] text-slate-400 leading-tight">
                            {qMeta.driver}
                          </div>
                        </div>

                        {/* Interactive Quantum Wire & Gates */}
                        <div className="relative flex-1 flex items-center h-14 bg-slate-900/60 rounded-xl px-4 border border-slate-800">
                          {/* Horizontal Wire Line */}
                          <div className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-[2px] bg-slate-700 z-0"></div>

                          {/* Gate Sequence */}
                          <div className="relative z-10 flex items-center gap-3.5 w-full">
                            {/* Initial Ground State */}
                            <span
                              title="Initial Ground State |0⟩"
                              className="text-[10px] font-mono font-bold text-slate-400 bg-slate-950 px-2 py-1 rounded border border-slate-800 shadow-xs"
                            >
                              |0⟩
                            </span>

                            {selectedFeatureMap === "zz_feature_map" ? (
                              <>
                                {/* Layer 1: Hadamard Superposition */}
                                <button
                                  type="button"
                                  onClick={() =>
                                    setSelectedGate({
                                      gate: "H",
                                      qubit: qIdx,
                                      desc: "Hadamard Transformation creating uniform superposition",
                                      unitary: "1/√2 * [[1, 1], [1, -1]]",
                                      param: null
                                    })
                                  }
                                  title="Hadamard Gate (Superposition)"
                                  className="w-8 h-8 rounded-lg bg-amber-950/90 border border-amber-500/80 text-amber-200 font-mono font-bold text-xs flex items-center justify-center shadow-md hover:scale-110 hover:border-amber-400 transition-all cursor-pointer"
                                >
                                  H
                                </button>

                                {/* Layer 2: Single-Qubit Phase Rotation RZ(2*theta) */}
                                <button
                                  type="button"
                                  onClick={() =>
                                    setSelectedGate({
                                      gate: "RZ",
                                      qubit: qIdx,
                                      desc: `Single-Qubit Phase Rotation RZ(2θ_${qIdx}) encoding feature amplitude`,
                                      unitary: `exp(-i * θ * Z) = [[e^(-i*${angle.toFixed(2)}), 0], [0, e^(i*${angle.toFixed(2)})]]`,
                                      param: 2.0 * angle
                                    })
                                  }
                                  title={`RZ Phase Gate: 2θ = ${twoTheta} rad`}
                                  className="px-2.5 py-1 rounded-lg bg-purple-950/90 border border-purple-500/80 text-purple-200 font-mono text-xs flex flex-col items-center justify-center shadow-md hover:scale-105 hover:border-purple-400 transition-all cursor-pointer"
                                >
                                  <span className="font-bold text-[11px]">RZ</span>
                                  <span className="text-[9px] text-purple-300 font-semibold">{twoTheta}</span>
                                </button>

                                {/* Layer 3: Entangling CNOT Interaction Ring */}
                                <div className="flex items-center gap-2 px-2 py-0.5 rounded bg-slate-950/70 border border-slate-800">
                                  <button
                                    type="button"
                                    onClick={() =>
                                      setSelectedGate({
                                        gate: "CNOT",
                                        qubit: qIdx,
                                        target: (qIdx + 1) % 4,
                                        desc: `Controlled-NOT entangling Qubit ${qIdx} with Qubit ${(qIdx + 1) % 4}`,
                                        unitary: "[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]",
                                        param: null
                                      })
                                    }
                                    title={`CNOT Entangler: Control on Q${qIdx}, Target on Q${(qIdx + 1) % 4}`}
                                    className="w-6 h-6 rounded-full bg-indigo-600 border border-indigo-400 text-white font-bold text-xs flex items-center justify-center shadow-sm hover:scale-110 transition-all cursor-pointer"
                                  >
                                    •
                                  </button>

                                  {/* Two-qubit interaction phase gate RZ(Phi) */}
                                  <button
                                    type="button"
                                    onClick={() =>
                                      setSelectedGate({
                                        gate: "RZ_ZZ",
                                        qubit: qIdx,
                                        target: (qIdx + 1) % 4,
                                        desc: `Two-Qubit Interaction Phase RZ(2*(π - θ_${qIdx})*(π - θ_${(qIdx + 1) % 4}))`,
                                        unitary: `exp(i * Φ_${qIdx}${(qIdx + 1) % 4} * Z ⊗ Z)`,
                                        param: entPhase ? entPhase.phi : null
                                      })
                                    }
                                    title={`Entanglement Phase Φ_${qIdx}${(qIdx + 1) % 4} = ${entPhase ? entPhase.phi.toFixed(2) : ""} rad`}
                                    className="px-2 py-0.5 rounded bg-indigo-950 border border-indigo-500/60 text-indigo-200 font-mono text-[10px] hover:border-indigo-400 transition-all cursor-pointer"
                                  >
                                    <span>Φ = {entPhase ? entPhase.phi.toFixed(2) : "0"}</span>
                                  </button>

                                  <button
                                    type="button"
                                    onClick={() =>
                                      setSelectedGate({
                                        gate: "CNOT Target",
                                        qubit: (qIdx + 3) % 4,
                                        target: qIdx,
                                        desc: `CNOT Target: Inverts Qubit ${qIdx} conditioned on Qubit ${(qIdx + 3) % 4}`,
                                        unitary: "Controlled Bit-Flip Unitary",
                                        param: null
                                      })
                                    }
                                    title={`CNOT Target from Qubit ${(qIdx + 3) % 4}`}
                                    className="w-6 h-6 rounded-full bg-indigo-500/80 border border-indigo-300 text-white font-bold text-xs flex items-center justify-center shadow-sm hover:scale-110 transition-all cursor-pointer"
                                  >
                                    ⊕
                                  </button>
                                </div>
                              </>
                            ) : (
                              /* Angle Embedding: Separable RY Rotations */
                              <button
                                type="button"
                                onClick={() =>
                                  setSelectedGate({
                                    gate: "RY",
                                    qubit: qIdx,
                                    desc: `Single-Qubit State Rotation RY(θ_${qIdx}) on Bloch sphere`,
                                    unitary: `[[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]`,
                                    param: angle
                                  })
                                }
                                title={`RY Rotation: ${angle.toFixed(3)} rad (${angleDeg}°)`}
                                className="px-3 py-1.5 rounded-lg bg-cyan-950/90 border border-cyan-500/80 text-cyan-200 font-mono text-xs flex items-center gap-1.5 shadow-md hover:scale-105 hover:border-cyan-400 transition-all cursor-pointer"
                              >
                                <span className="font-bold">RY</span>
                                <span className="text-[11px] font-semibold">{angle.toFixed(3)} rad</span>
                              </button>
                            )}

                            {/* Measurement Terminal */}
                            <div className="ml-auto flex items-center gap-1.5 text-xs text-emerald-400 bg-slate-950 border border-slate-800 px-3 py-1 rounded-lg font-mono">
                              <Activity className="w-3.5 h-3.5" />
                              <span>|ψ⟩ Wire {qIdx}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Gate Inspector Tooltip / Box */}
              {selectedGate && (
                <div className="mt-5 p-4 rounded-xl bg-slate-900 border border-indigo-500/50 shadow-md flex flex-col sm:flex-row sm:items-center justify-between gap-4 animate-in fade-in">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-indigo-600 text-white">
                        {selectedGate.gate} GATE
                      </span>
                      <span className="text-xs font-semibold text-white">
                        Target: Qubit |q_{selectedGate.qubit}⟩
                        {selectedGate.target !== undefined && selectedGate.target !== null && ` ↔ Qubit |q_${selectedGate.target}⟩`}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300">{selectedGate.desc}</p>
                    <p className="text-[11px] font-mono text-indigo-300">Unitary: {selectedGate.unitary}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSelectedGate(null)}
                    className="self-end sm:self-center text-xs text-slate-400 hover:text-white px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 cursor-pointer"
                  >
                    Dismiss
                  </button>
                </div>
              )}

              {/* Circuit Legend & Math */}
              <div className="mt-5 pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-xs text-slate-400 gap-4">
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded bg-amber-900 border border-amber-500 inline-block"></span>
                    <span>H: Superposition Layer</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded bg-purple-900 border border-purple-500 inline-block"></span>
                    <span>RZ(2θ): Feature Phase</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-full bg-indigo-600 inline-block"></span>
                    <span>• / ⊕: CNOT Entangler</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded bg-cyan-900 border border-cyan-500 inline-block"></span>
                    <span>RY: Angle Rotation</span>
                  </div>
                </div>

                <div className="text-[11px] font-mono text-indigo-300">
                  U_Φ(x) = exp(i ∑ θ_i Z_i + i ∑ Φ_ij Z_i Z_j)
                </div>
              </div>
            </div>

            {/* Step-by-Step State Evolution Stages */}
            <div className="space-y-3 pt-2">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                <Binary className="w-4 h-4 text-indigo-600" />
                <span>Quantum State Evolution in 16-Dimensional Hilbert Space</span>
              </h4>

              <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-1.5">
                  <div className="font-bold text-slate-900 flex items-center justify-between">
                    <span>1. Ground State</span>
                    <span className="text-[10px] text-indigo-600 font-mono">Stage 0</span>
                  </div>
                  <div className="font-mono text-[11px] text-slate-700 bg-white p-1.5 rounded border border-slate-200">
                    |0000⟩ = [1, 0, ..., 0]ᵀ
                  </div>
                  <p className="text-[10px] text-slate-500">
                    Qubits initialized to standard vacuum state before clinical feature injection.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-1.5">
                  <div className="font-bold text-slate-900 flex items-center justify-between">
                    <span>2. Superposition</span>
                    <span className="text-[10px] text-indigo-600 font-mono">Stage 1</span>
                  </div>
                  <div className="font-mono text-[11px] text-slate-700 bg-white p-1.5 rounded border border-slate-200">
                    H^⊗4|0000⟩ = 1/4 ∑ |k⟩
                  </div>
                  <p className="text-[10px] text-slate-500">
                    Hadamard layer spreads amplitude equally over all 16 computational basis states.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-1.5">
                  <div className="font-bold text-slate-900 flex items-center justify-between">
                    <span>3. Feature Phase</span>
                    <span className="text-[10px] text-indigo-600 font-mono">Stage 2</span>
                  </div>
                  <div className="font-mono text-[11px] text-slate-700 bg-white p-1.5 rounded border border-slate-200">
                    RZ(2θ_i) = e^(-i θ_i Z_i)
                  </div>
                  <p className="text-[10px] text-slate-500">
                    Encodes standardized patient biomarkers directly into quantum phases along Z-axis.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-1.5">
                  <div className="font-bold text-slate-900 flex items-center justify-between">
                    <span>4. Entanglement</span>
                    <span className="text-[10px] text-indigo-600 font-mono">Stage 3</span>
                  </div>
                  <div className="font-mono text-[11px] text-slate-700 bg-white p-1.5 rounded border border-slate-200">
                    e^(i Φ_ij Z_i Z_j)
                  </div>
                  <p className="text-[10px] text-slate-500">
                    Circular CNOTs create non-local quantum correlations impossible in classical linear SVMs.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-1.5">
                  <div className="font-bold text-slate-900 flex items-center justify-between">
                    <span>5. Kernel Fidelity</span>
                    <span className="text-[10px] text-indigo-600 font-mono">Stage 4</span>
                  </div>
                  <div className="font-mono text-[11px] text-slate-700 bg-white p-1.5 rounded border border-slate-200">
                    K(x, x') = |⟨ψ|ψ'⟩|²
                  </div>
                  <p className="text-[10px] text-slate-500">
                    Statevector inner product evaluates fidelity kernel used by support vector classifier.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* SECTION 2: QUANTUM HEATMAPS */}
      {(activeSection === "all" || activeSection === "heatmaps") && (
        <section id="section-heatmaps" className="space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
            {/* Heatmaps Header & Sub-Tabs */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-5">
              <div>
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-indigo-50 border border-indigo-200 text-indigo-700 flex items-center justify-center font-bold text-xs">
                    <Grid className="w-4 h-4" />
                  </div>
                  <h3 className="text-base font-bold text-slate-900">
                    Quantum Mechanical & Hilbert Space Heatmaps
                  </h3>
                </div>
                <p className="text-xs text-slate-500 mt-1">
                  Empirical transition fidelity Gram matrices and clinical biomarker loading weight distributions
                </p>
              </div>

              {/* Heatmap Sub-tabs */}
              <div className="inline-flex rounded-lg bg-slate-100 p-0.5 border border-slate-200">
                <button
                  id="hmap-tab-gram"
                  onClick={() => setHeatmapTab("gram")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer flex items-center gap-1.5 ${
                    heatmapTab === "gram"
                      ? "bg-white text-indigo-700 font-bold shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Grid className="w-3.5 h-3.5" />
                  <span>Gram Matrix [K(x_i, x_j)]</span>
                </button>

                <button
                  id="hmap-tab-pca"
                  onClick={() => setHeatmapTab("pca")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer flex items-center gap-1.5 ${
                    heatmapTab === "pca"
                      ? "bg-white text-indigo-700 font-bold shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <BarChart3 className="w-3.5 h-3.5" />
                  <span>Clinical-to-Qubit Loading</span>
                </button>

                <button
                  id="hmap-tab-fidelities"
                  onClick={() => setHeatmapTab("fidelities")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all cursor-pointer flex items-center gap-1.5 ${
                    heatmapTab === "fidelities"
                      ? "bg-white text-indigo-700 font-bold shadow-xs"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" />
                  <span>Qubit Correlation Spectrum</span>
                </button>
              </div>
            </div>

            {/* TAB A: QUANTUM KERNEL GRAM MATRIX HEATMAP */}
            {heatmapTab === "gram" && (
              <div className="space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                      <span>PennyLane State Fidelity Gram Matrix</span>
                      <span className="font-mono text-xs px-2 py-0.5 rounded bg-indigo-50 border border-indigo-200 text-indigo-700">
                        {gramMatrix.length}×{gramMatrix.length} Samples
                      </span>
                    </h4>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Pairwise quantum state overlap: $K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$ computed via PennyLane statevectors
                    </p>
                  </div>

                  {/* Filter selector */}
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-500">Highlight:</span>
                    <select
                      id="gram-filter-select"
                      value={gramFilter}
                      onChange={(e) => setGramFilter(e.target.value as any)}
                      className="px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-xs font-medium text-slate-700 cursor-pointer"
                    >
                      <option value="all">All 25×25 State Pairs</option>
                      <option value="diagonal">Diagonal Self-Overlap (K = 1.0)</option>
                      <option value="high">High State Fidelity (K &gt; 0.5)</option>
                      <option value="off_diagonal">Off-Diagonal Cross-Correlations</option>
                    </select>
                  </div>
                </div>

                {/* Selected Cell Banner */}
                {selectedGramCell && (
                  <div className="p-3 rounded-xl bg-indigo-50 border border-indigo-200 text-xs flex flex-wrap items-center justify-between gap-3 animate-in fade-in">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-indigo-900">
                        Inspecting Pair: Patient {selectedGramCell.row + 1} ↔ Patient {selectedGramCell.col + 1}
                      </span>
                      <span className="font-mono text-indigo-700 font-bold px-2 py-0.5 rounded bg-white border border-indigo-200">
                        K(x_{selectedGramCell.row}, x_{selectedGramCell.col}) = {selectedGramCell.val.toFixed(4)}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-600">
                      {selectedGramCell.row === selectedGramCell.col
                        ? "Identical Quantum State: Self-fidelity normalized to exactly 1.000"
                        : selectedGramCell.val > 0.6
                        ? "High Quantum Hilbert Overlap: Strong clinical similarity in ischemic feature space"
                        : selectedGramCell.val > 0.25
                        ? "Moderate Quantum Coupling: Partial overlap across orthogonal eigenstates"
                        : "Nearly Orthogonal States: Distinct cardiovascular diagnostic profiles"}
                    </div>
                  </div>
                )}

                {/* The 25x25 Matrix Grid */}
                {gramMatrix.length > 0 ? (
                  <div className="overflow-x-auto pb-2">
                    <div className="inline-block border border-slate-300 rounded-xl p-3 bg-slate-900 shadow-md">
                      <div
                        className="grid gap-[2.5px]"
                        style={{
                          gridTemplateColumns: `repeat(${gramMatrix.length}, minmax(18px, 1fr))`
                        }}
                      >
                        {gramMatrix.map((row, rIdx) =>
                          row.map((val, cIdx) => {
                            const isDiagonal = rIdx === cIdx;
                            const isHigh = val > 0.5;

                            let shouldHighlight = true;
                            if (gramFilter === "diagonal" && !isDiagonal) shouldHighlight = false;
                            if (gramFilter === "high" && !isHigh) shouldHighlight = false;
                            if (gramFilter === "off_diagonal" && isDiagonal) shouldHighlight = false;

                            // Color scale for fidelity [0, 1]
                            const bg =
                              val > 0.85
                                ? "bg-indigo-500 text-white"
                                : val > 0.6
                                ? "bg-indigo-600/90 text-white"
                                : val > 0.4
                                ? "bg-indigo-800/80 text-indigo-100"
                                : val > 0.2
                                ? "bg-slate-800 text-slate-300"
                                : "bg-slate-950 text-slate-500 border border-slate-800/60";

                            const isSelected =
                              selectedGramCell?.row === rIdx && selectedGramCell?.col === cIdx;

                            return (
                              <div
                                key={`${rIdx}-${cIdx}`}
                                onMouseEnter={() => setSelectedGramCell({ row: rIdx, col: cIdx, val })}
                                onClick={() => setSelectedGramCell({ row: rIdx, col: cIdx, val })}
                                title={`Sample ${rIdx + 1} ↔ Sample ${cIdx + 1} | Fidelity: ${val.toFixed(4)}`}
                                className={`w-4 h-4 sm:w-5 sm:h-5 rounded-[2px] transition-all cursor-pointer flex items-center justify-center text-[7px] sm:text-[8px] font-mono ${bg} ${
                                  !shouldHighlight ? "opacity-25" : "opacity-100"
                                } ${
                                  isSelected
                                    ? "ring-2 ring-amber-400 z-20 scale-125 shadow-lg"
                                    : "hover:scale-110 hover:ring-1 hover:ring-indigo-300"
                                }`}
                              >
                                {isDiagonal ? "1" : ""}
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-8 text-center text-xs text-slate-500 bg-slate-50 rounded-xl border">
                    Loading Quantum Kernel Gram Matrix...
                  </div>
                )}

                {/* Gram Heatmap Legend & Diagnostics */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-500 pt-3 border-t border-slate-100 gap-4">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="font-semibold text-slate-700">Fidelity Scale:</span>
                    <div className="flex items-center gap-1 text-[11px]">
                      <span className="w-3 h-3 rounded-[2px] bg-slate-950 border border-slate-700 inline-block"></span>
                      <span>0.0 (Orthogonal)</span>
                    </div>
                    <div className="flex items-center gap-1 text-[11px]">
                      <span className="w-3 h-3 rounded-[2px] bg-slate-800 inline-block"></span>
                      <span>0.25 (Low)</span>
                    </div>
                    <div className="flex items-center gap-1 text-[11px]">
                      <span className="w-3 h-3 rounded-[2px] bg-indigo-800 inline-block"></span>
                      <span>0.50 (Moderate)</span>
                    </div>
                    <div className="flex items-center gap-1 text-[11px]">
                      <span className="w-3 h-3 rounded-[2px] bg-indigo-500 inline-block"></span>
                      <span>1.0 (Identical)</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 text-[11px] font-mono">
                    <span className="text-slate-600">Mean Off-Diagonal:</span>
                    <span className="font-bold text-indigo-600">{gramStats.meanFidelity.toFixed(3)}</span>
                    <span className="text-slate-300">|</span>
                    <span className="text-slate-600">Max Correlation:</span>
                    <span className="font-bold text-emerald-600">{gramStats.maxOffDiag.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* TAB B: CLINICAL-TO-QUBIT LOADING HEATMAP (PCA WEIGHT MATRIX) */}
            {heatmapTab === "pca" && (
              <div className="space-y-5">
                <div>
                  <h4 className="text-sm font-bold text-slate-900">
                    Clinical Biomarker to Quantum Qubit Projection Heatmap
                  </h4>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Heatmap matrix showing how 6 cardiovascular risk factors are mathematically weighted onto the 4 Qubits
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50 text-slate-700 font-semibold">
                        <th className="p-3">Quantum Wire / Qubit</th>
                        <th className="p-3">Explained Variance</th>
                        {FEATURE_NAMES.map((f, i) => (
                          <th key={i} className="p-3 text-center">
                            {f}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {QUBIT_METADATA.map((qMeta, qIdx) => {
                        const weights = PCA_COMPONENTS[qIdx];
                        const variance = PCA_VARIANCE[qIdx];

                        return (
                          <tr key={qIdx} className="hover:bg-slate-50/80 transition-colors">
                            <td className="p-3">
                              <div className="font-mono font-bold text-indigo-700">{qMeta.name}</div>
                              <div className="text-[10px] text-slate-500 font-sans">{qMeta.title}</div>
                            </td>
                            <td className="p-3 font-mono font-bold text-slate-800">
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                                {variance.toFixed(1)}%
                              </span>
                            </td>
                            {weights.map((w, wIdx) => {
                              const absW = Math.abs(w);
                              // Color intensity based on loading magnitude
                              const isPositive = w >= 0;
                              const bg =
                                absW > 0.5
                                  ? isPositive
                                    ? "bg-indigo-600 text-white font-bold"
                                    : "bg-rose-600 text-white font-bold"
                                  : absW > 0.25
                                  ? isPositive
                                    ? "bg-indigo-100 text-indigo-900 font-semibold"
                                    : "bg-rose-100 text-rose-900 font-semibold"
                                  : "bg-slate-50 text-slate-600";

                              return (
                                <td key={wIdx} className="p-2 text-center">
                                  <div
                                    title={`Qubit ${qIdx} loading on ${FEATURE_NAMES[wIdx]}: ${w.toFixed(4)}`}
                                    className={`py-2 px-2.5 rounded-md font-mono text-[11px] transition-all ${bg}`}
                                  >
                                    {w > 0 ? `+${w.toFixed(3)}` : w.toFixed(3)}
                                  </div>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="flex flex-wrap items-center justify-between text-xs text-slate-500 pt-3 border-t border-slate-100 gap-4">
                  <div className="flex items-center gap-4">
                    <span className="font-semibold text-slate-700">Loading Direction:</span>
                    <div className="flex items-center gap-1 text-[11px]">
                      <span className="w-3 h-3 rounded bg-indigo-600 inline-block"></span>
                      <span>Strong Positive (+0.50 to +1.0)</span>
                    </div>
                    <div className="flex items-center gap-1 text-[11px]">
                      <span className="w-3 h-3 rounded bg-rose-600 inline-block"></span>
                      <span>Strong Negative (-0.50 to -1.0)</span>
                    </div>
                    <div className="flex items-center gap-1 text-[11px]">
                      <span className="w-3 h-3 rounded bg-slate-100 border inline-block"></span>
                      <span>Neutral Baseline (&lt; ±0.25)</span>
                    </div>
                  </div>

                  <span className="font-mono text-[11px] text-slate-600">
                    Cumulative 4-Qubit Explained Variance: 86.84%
                  </span>
                </div>
              </div>
            )}

            {/* TAB C: QUBIT CORRELATION SPECTRUM */}
            {heatmapTab === "fidelities" && (
              <div className="space-y-5">
                <div>
                  <h4 className="text-sm font-bold text-slate-900">
                    Active Patient Qubit Correlation Spectrum
                  </h4>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Pairwise angular distance Δθ_ij = |θ_i - θ_j| and circular entangling phase strengths
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {QUBIT_METADATA.map((qMeta, i) => {
                    const angleI = qubitAngles[i];
                    return (
                      <div key={i} className="p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="font-mono font-bold text-indigo-700 text-xs">{qMeta.name}</span>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-200 font-semibold text-slate-800">
                            {angleI.toFixed(3)} rad
                          </span>
                        </div>
                        <div className="text-xs font-semibold text-slate-900">{qMeta.title}</div>

                        <div className="space-y-1.5 pt-1">
                          <span className="text-[10px] font-semibold text-slate-500 uppercase">Coupling with Other Wires:</span>
                          {[0, 1, 2, 3].filter((j) => j !== i).map((j) => {
                            const angleJ = qubitAngles[j];
                            const diff = Math.abs(angleI - angleJ);
                            const coupling = Math.sin(diff / 2) ** 2;

                            return (
                              <div key={j} className="flex items-center justify-between text-[11px] font-mono bg-white p-1.5 rounded border border-slate-200">
                                <span className="text-slate-600">↔ Wire |q_{String(j)}⟩:</span>
                                <span className="font-bold text-indigo-600">{coupling.toFixed(3)}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="p-4 rounded-xl bg-indigo-50/70 border border-indigo-200 text-xs text-indigo-900 flex items-start gap-2.5">
                  <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
                  <p className="leading-relaxed">
                    In the Havlíček ZZ-feature map, two-qubit correlations exp(i Φ_ij Z_i Z_j) map
                    multi-factorial clinical interactions (such as the combined effect of high serum cholesterol and
                    ST depression under exercise stress) onto entangling phases that quantum kernel SVMs exploit
                    for superior non-linear decision margins.
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
};
