import React from "react";
import { PredictionResult, ClinicalInput } from "../types";
import {
  AlertTriangle,
  CheckCircle,
  Activity,
  Layers,
  Sparkles,
  Info,
  Compass,
  ArrowRight,
  ShieldAlert,
  ShieldCheck
} from "lucide-react";

interface ResultCardProps {
  result: PredictionResult;
  patientInput?: ClinicalInput;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, patientInput }) => {
  const isPositive = result.prediction === 1 || result.cad_status === "POSITIVE";
  const probPercent = (result.chd_probability * 100).toFixed(1);
  const cadLabel = result.cad_diagnosis || (isPositive ? "CAD Positive" : "CAD Negative");

  // Clinical reference helpers
  const ageVal = patientInput?.age ?? 54;
  const sexVal = patientInput?.sex ?? 1;
  const cholVal = patientInput?.cholesterol ?? 200;
  const bpVal = patientInput?.resting_bp ?? 120;
  const anginaVal = patientInput?.exercise_angina ?? 0;
  const stVal = patientInput?.st_depression ?? 0.0;

  const ageStatus =
    ageVal < 45
      ? { label: "Baseline (< 45)", color: "emerald", risk: false }
      : ageVal < 60
      ? { label: "Moderate (45-59)", color: "amber", risk: false }
      : { label: "Elevated Risk (≥ 60)", color: "rose", risk: true };

  const sexStatus =
    sexVal === 1
      ? { label: "Male (Value 1)", color: "indigo", risk: false }
      : { label: "Female (Value 0)", color: "purple", risk: false };

  const cholStatus =
    cholVal < 200
      ? { label: "Desirable (Normal)", color: "emerald", risk: false }
      : cholVal < 240
      ? { label: "Borderline High", color: "amber", risk: true }
      : { label: "High (CAD Risk Factor)", color: "rose", risk: true };

  const bpStatus =
    bpVal < 120
      ? { label: "Normal (<120)", color: "emerald", risk: false }
      : bpVal < 130
      ? { label: "Elevated (120-129)", color: "amber", risk: true }
      : { label: "Hypertension (≥130)", color: "rose", risk: true };

  const anginaStatus =
    anginaVal === 0
      ? { label: "Absent (Normal)", color: "emerald", risk: false }
      : { label: "Present (CAD Symptom)", color: "rose", risk: true };

  const stStatus =
    stVal < 1.0
      ? { label: "Normal (<1.0 mm)", color: "emerald", risk: false }
      : stVal < 2.0
      ? { label: "Equivocal (1.0-1.9 mm)", color: "amber", risk: true }
      : { label: "Significant Ischemia (≥2.0 mm)", color: "rose", risk: true };

  const riskFactorsCount = [cholStatus.risk, bpStatus.risk, anginaStatus.risk, stStatus.risk, ageStatus.risk].filter(Boolean).length;

  return (
    <div id="quantum-prediction-result" className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
      {/* Primary CAD Diagnostic Banner */}
      <div
        className={`p-6 rounded-2xl border-2 flex flex-wrap items-center justify-between gap-5 transition-all shadow-sm ${
          isPositive
            ? "bg-rose-50/90 border-rose-300 text-rose-950"
            : "bg-emerald-50/90 border-emerald-300 text-emerald-950"
        }`}
      >
        <div className="flex items-center gap-4">
          <div
            className={`w-16 h-16 rounded-2xl flex items-center justify-center shrink-0 shadow-md ${
              isPositive ? "bg-rose-600 text-white" : "bg-emerald-600 text-white"
            }`}
          >
            {isPositive ? <AlertTriangle className="w-9 h-9" /> : <CheckCircle className="w-9 h-9" />}
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[11px] font-bold tracking-wider uppercase opacity-75">
                Quantum Model Diagnosis
              </span>
              <span
                className={`text-[11px] px-2.5 py-0.5 rounded-full font-extrabold border ${
                  isPositive
                    ? "bg-rose-200 text-rose-900 border-rose-400"
                    : "bg-emerald-200 text-emerald-900 border-emerald-400"
                }`}
              >
                {result.risk_level} Risk Tier
              </span>
            </div>

            <div className="text-3xl sm:text-4xl font-black tracking-tight flex items-center gap-3">
              <span className={isPositive ? "text-rose-700" : "text-emerald-700"}>
                {cadLabel}
              </span>
            </div>

            <div className="text-xs text-slate-700 mt-1.5 font-medium flex items-center gap-1.5">
              {isPositive ? (
                <>
                  <ShieldAlert className="w-4 h-4 text-rose-600 shrink-0" />
                  <span>
                    <strong>Coronary Artery Disease Detected:</strong> High probability of myocardial ischemia based on input biomarkers.
                  </span>
                </>
              ) : (
                <>
                  <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
                  <span>
                    <strong>Coronary Artery Disease Ruled Out:</strong> Biomarkers align with healthy coronary reference standards.
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Probability Gauge Box */}
        <div className="text-right sm:border-l sm:border-slate-200 sm:pl-6 bg-white/70 px-4 py-3 rounded-xl border border-slate-200/60 shadow-xs">
          <div className="text-xs text-slate-500 font-medium">Confidence Probability</div>
          <div className="text-3xl font-black font-mono tracking-tight text-slate-900">
            {probPercent}%
          </div>
          <div className="text-[11px] text-slate-500 font-mono mt-0.5">
            SVM Margin: {result.decision_margin > 0 ? `+${result.decision_margin.toFixed(3)}` : result.decision_margin.toFixed(3)}
          </div>
        </div>
      </div>

      {/* Input Data vs. Clinical Reference Comparison Table */}
      {patientInput && (
        <div className="border border-slate-200 rounded-xl overflow-hidden shadow-xs">
          <div className="p-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-600" />
              <h3 className="text-xs font-bold text-slate-900">
                Input Clinical Data vs. Reference Standards
              </h3>
            </div>
            <span className="text-[11px] font-medium text-slate-500">
              {riskFactorsCount} of 6 parameters indicate risk
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-slate-100/70 text-slate-600 font-semibold border-b border-slate-200">
                <tr>
                  <th className="p-3">Clinical Parameter</th>
                  <th className="p-3">Your Entered Value</th>
                  <th className="p-3">Standard Reference Range</th>
                  <th className="p-3">Reference Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {/* 1. Age */}
                <tr className={ageStatus.risk ? "bg-rose-50/20" : ""}>
                  <td className="p-3 font-semibold text-slate-900">
                    Patient Age
                    <span className="block text-[10px] font-normal text-slate-500">Demographic risk</span>
                  </td>
                  <td className="p-3 font-mono font-bold text-slate-900">
                    {ageVal} <span className="text-slate-500 font-normal">years</span>
                  </td>
                  <td className="p-3 text-slate-600">
                    &lt; 45 yrs (Lower Risk Baseline)
                  </td>
                  <td className="p-3">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${
                        ageStatus.color === "emerald"
                          ? "bg-emerald-100 text-emerald-800 border-emerald-300"
                          : ageStatus.color === "amber"
                          ? "bg-amber-100 text-amber-800 border-amber-300"
                          : "bg-rose-100 text-rose-800 border-rose-300"
                      }`}
                    >
                      {ageStatus.label}
                    </span>
                  </td>
                </tr>

                {/* 2. Biological Sex */}
                <tr>
                  <td className="p-3 font-semibold text-slate-900">
                    Biological Sex
                    <span className="block text-[10px] font-normal text-slate-500">Cardiovascular epidemiology</span>
                  </td>
                  <td className="p-3 font-mono font-bold text-slate-900">
                    {sexVal === 1 ? "Male (1)" : "Female (0)"}
                  </td>
                  <td className="p-3 text-slate-600">
                    1 = Male | 0 = Female
                  </td>
                  <td className="p-3">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${
                        sexVal === 1
                          ? "bg-indigo-100 text-indigo-800 border-indigo-300"
                          : "bg-purple-100 text-purple-800 border-purple-300"
                      }`}
                    >
                      {sexStatus.label}
                    </span>
                  </td>
                </tr>

                {/* 3. Serum Cholesterol */}
                <tr className={cholStatus.risk ? "bg-amber-50/20" : ""}>
                  <td className="p-3 font-semibold text-slate-900">
                    Serum Cholesterol
                    <span className="block text-[10px] font-normal text-slate-500">Lipid profile</span>
                  </td>
                  <td className="p-3 font-mono font-bold text-slate-900">
                    {cholVal} <span className="text-slate-500 font-normal">mg/dL</span>
                  </td>
                  <td className="p-3 text-slate-600">
                    &lt; 200 mg/dL <span className="text-[10px] text-slate-400">(Desirable)</span>
                  </td>
                  <td className="p-3">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${
                        cholStatus.color === "emerald"
                          ? "bg-emerald-100 text-emerald-800 border-emerald-300"
                          : cholStatus.color === "amber"
                          ? "bg-amber-100 text-amber-800 border-amber-300"
                          : "bg-rose-100 text-rose-800 border-rose-300"
                      }`}
                    >
                      {cholStatus.label}
                    </span>
                  </td>
                </tr>

                {/* 2. Resting Blood Pressure */}
                <tr className={bpStatus.risk ? "bg-amber-50/20" : ""}>
                  <td className="p-3 font-semibold text-slate-900">
                    Resting Blood Pressure
                    <span className="block text-[10px] font-normal text-slate-500">Systolic pressure</span>
                  </td>
                  <td className="p-3 font-mono font-bold text-slate-900">
                    {bpVal} <span className="text-slate-500 font-normal">mm Hg</span>
                  </td>
                  <td className="p-3 text-slate-600">
                    &lt; 120 mm Hg <span className="text-[10px] text-slate-400">(Normal)</span>
                  </td>
                  <td className="p-3">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${
                        bpStatus.color === "emerald"
                          ? "bg-emerald-100 text-emerald-800 border-emerald-300"
                          : bpStatus.color === "amber"
                          ? "bg-amber-100 text-amber-800 border-amber-300"
                          : "bg-rose-100 text-rose-800 border-rose-300"
                      }`}
                    >
                      {bpStatus.label}
                    </span>
                  </td>
                </tr>

                {/* 3. Angina */}
                <tr className={anginaStatus.risk ? "bg-rose-50/20" : ""}>
                  <td className="p-3 font-semibold text-slate-900">
                    Exercise-Induced Angina
                    <span className="block text-[10px] font-normal text-slate-500">Exertional chest pain</span>
                  </td>
                  <td className="p-3 font-mono font-bold text-slate-900">
                    {anginaVal === 1 ? (
                      <span className="text-rose-700">1: Present (Yes)</span>
                    ) : (
                      <span className="text-emerald-700">0: Absent (No)</span>
                    )}
                  </td>
                  <td className="p-3 text-slate-600">
                    Absent (No chest discomfort)
                  </td>
                  <td className="p-3">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${
                        anginaStatus.color === "emerald"
                          ? "bg-emerald-100 text-emerald-800 border-emerald-300"
                          : "bg-rose-100 text-rose-800 border-rose-300"
                      }`}
                    >
                      {anginaStatus.label}
                    </span>
                  </td>
                </tr>

                {/* 4. ST Segment Depression */}
                <tr className={stStatus.risk ? "bg-amber-50/20" : ""}>
                  <td className="p-3 font-semibold text-slate-900">
                    ST Segment Depression
                    <span className="block text-[10px] font-normal text-slate-500">ECG Oldpeak</span>
                  </td>
                  <td className="p-3 font-mono font-bold text-slate-900">
                    {stVal.toFixed(1)} <span className="text-slate-500 font-normal">mm</span>
                  </td>
                  <td className="p-3 text-slate-600">
                    &lt; 1.0 mm <span className="text-[10px] text-slate-400">(Isoelectric)</span>
                  </td>
                  <td className="p-3">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${
                        stStatus.color === "emerald"
                          ? "bg-emerald-100 text-emerald-800 border-emerald-300"
                          : stStatus.color === "amber"
                          ? "bg-amber-100 text-amber-800 border-amber-300"
                          : "bg-rose-100 text-rose-800 border-rose-300"
                      }`}
                    >
                      {stStatus.label}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="p-3 bg-slate-50/80 border-t border-slate-200 text-xs text-slate-700">
            <strong>Reference Conclusion:</strong>{" "}
            {isPositive ? (
              <span>
                Based on the input values, the patient shows {riskFactorsCount} key risk factors exceeding reference thresholds. The PennyLane QSVC quantum kernel evaluated the 4-qubit Hilbert state and produced a positive decision boundary margin, concluding <strong>CAD Positive</strong>.
              </span>
            ) : (
              <span>
                Based on the input values, the clinical markers remain within or near desirable reference bounds. The PennyLane QSVC quantum kernel mapped the state into the negative decision boundary half-space, concluding <strong>CAD Negative</strong>.
              </span>
            )}
          </div>
        </div>
      )}

      {/* Probability bar */}
      <div>
        <div className="flex items-center justify-between text-xs text-slate-600 mb-1.5 font-medium">
          <span>Ischemia Likelihood Gradient</span>
          <span className="font-mono text-slate-900 font-bold">{probPercent}%</span>
        </div>
        <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden flex">
          <div
            className={`h-full transition-all duration-700 ease-out ${
              isPositive ? "bg-rose-500" : "bg-emerald-500"
            }`}
            style={{ width: `${Math.max(4, Math.min(99, Number(probPercent)))}%` }}
          ></div>
        </div>
        <div className="flex justify-between text-[10px] text-slate-400 mt-1">
          <span>0% (Negligible)</span>
          <span>50% (Equivocal)</span>
          <span>100% (Confirmed Ischemia)</span>
        </div>
      </div>

      {/* Clinical Interpretation */}
      <div className="p-3.5 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-700 leading-relaxed flex items-start gap-2.5">
        <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-900">Clinical Cardiology Interpretation: </span>
          <span>{result.clinical_interpretation}</span>
        </div>
      </div>

      {/* 4-Qubit Quantum Bloch Rotation Angles */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-800">
            <Compass className="w-4 h-4 text-indigo-600" />
            <span>4-Qubit State Space Embeddings [θ₀, θ₁, θ₂, θ₃]</span>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            Domain: [0, π] rad
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {result.quantum_encoding.qubit_mapping.map((qm) => {
            const rad = qm.angle_rad;
            const deg = ((rad / Math.PI) * 180).toFixed(0);
            const percent = ((rad / Math.PI) * 100).toFixed(0);

            return (
              <div key={qm.qubit} className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div className="flex items-center justify-between text-xs font-mono font-bold text-indigo-700 mb-1">
                  <span>|q_{qm.qubit}⟩</span>
                  <span className="text-slate-500 text-[10px]">{deg}°</span>
                </div>
                <div className="text-[11px] font-medium text-slate-800 truncate mb-1">
                  {qm.feature}
                </div>
                <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-1.5">
                  <div
                    className="bg-indigo-600 h-full rounded-full"
                    style={{ width: `${percent}%` }}
                  ></div>
                </div>
                <div className="text-[10px] text-slate-400 font-mono mt-1">
                  {rad.toFixed(3)} rad
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Support Vector Fidelities */}
      {result.support_vector_fidelities && result.support_vector_fidelities.length > 0 && (
        <div className="border-t border-slate-100 pt-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-800">
              <Layers className="w-4 h-4 text-purple-600" />
              <span>Quantum Transition Fidelities With Support Vectors</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500">
              K(x, x_sv) = |⟨ψ(x)|ψ(x_sv)⟩|²
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {result.support_vector_fidelities.slice(0, 4).map((sv) => (
              <div
                key={sv.sv_index}
                className="p-2.5 rounded-lg border border-slate-200 bg-slate-50/50 flex flex-col justify-between"
              >
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-mono text-slate-600">SV #{sv.sv_index}</span>
                  <span
                    className={`text-[9px] px-1.5 py-0.2 rounded font-bold ${
                      sv.sv_class === "High Risk"
                        ? "bg-rose-100 text-rose-700"
                        : "bg-emerald-100 text-emerald-700"
                    }`}
                  >
                    {sv.sv_class}
                  </span>
                </div>
                <div className="mt-2 flex items-baseline justify-between font-mono">
                  <span className="text-[10px] text-slate-400">Overlap:</span>
                  <span className="text-xs font-bold text-slate-900">
                    {(sv.fidelity * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
