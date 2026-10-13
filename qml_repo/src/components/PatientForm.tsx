import React from "react";
import { ClinicalInput } from "../types";
import { Activity, Sparkles, Stethoscope, Play, CheckCircle2, AlertTriangle, XCircle, Info, User, UserCheck } from "lucide-react";

interface PatientFormProps {
  input: ClinicalInput;
  onChange: (updated: Partial<ClinicalInput>) => void;
  onSubmit: (e: React.FormEvent) => void;
  loading: boolean;
}

export const PatientForm: React.FC<PatientFormProps> = ({
  input,
  onChange,
  onSubmit,
  loading
}) => {
  const PRESETS = [
    {
      label: "Low Risk Sample (CAD Negative)",
      desc: "Age 34, Female, normal lipids & BP, no angina, flat ST",
      values: { age: 34, sex: 0, cholesterol: 172, resting_bp: 112, exercise_angina: 0, st_depression: 0.1 }
    },
    {
      label: "Borderline Risk Sample",
      desc: "Age 52, Male, borderline lipids (226), pre-HTN (128), 0 angina",
      values: { age: 52, sex: 1, cholesterol: 226, resting_bp: 128, exercise_angina: 0, st_depression: 0.8 }
    },
    {
      label: "High Risk Sample (CAD Positive)",
      desc: "Age 62, Male, high BP (150), Angina present, ST depression 2.5mm",
      values: { age: 62, sex: 1, cholesterol: 275, resting_bp: 150, exercise_angina: 1, st_depression: 2.5 }
    },
    {
      label: "Acute Ischemic Sample (CAD Positive)",
      desc: "Age 68, Male, severe ST depression 3.4mm with exertional angina",
      values: { age: 68, sex: 1, cholesterol: 255, resting_bp: 144, exercise_angina: 1, st_depression: 3.4 }
    }
  ];

  // Helper references for badges
  const ageStatus =
    input.age < 45
      ? { label: "Lower Baseline (< 45)", color: "emerald", icon: CheckCircle2 }
      : input.age < 60
      ? { label: "Moderate Demographic (45–59)", color: "amber", icon: AlertTriangle }
      : { label: "Elevated Age Risk (≥ 60)", color: "rose", icon: XCircle };

  const sexStatus =
    input.sex === 1
      ? { label: "Male (Value 1)", color: "indigo" }
      : { label: "Female (Value 0)", color: "purple" };

  const cholStatus =
    input.cholesterol < 200
      ? { label: "Desirable (< 200)", color: "emerald", icon: CheckCircle2 }
      : input.cholesterol < 240
      ? { label: "Borderline (200-239)", color: "amber", icon: AlertTriangle }
      : { label: "High Risk (≥ 240)", color: "rose", icon: XCircle };

  const bpStatus =
    input.resting_bp < 120
      ? { label: "Normal (< 120)", color: "emerald", icon: CheckCircle2 }
      : input.resting_bp < 130
      ? { label: "Elevated (120-129)", color: "amber", icon: AlertTriangle }
      : { label: "Hypertensive (≥ 130)", color: "rose", icon: XCircle };

  const anginaStatus =
    input.exercise_angina === 0
      ? { label: "0: Absent (Normal)", color: "emerald", icon: CheckCircle2 }
      : { label: "1: Present (CAD Risk)", color: "rose", icon: XCircle };

  const stStatus =
    input.st_depression < 1.0
      ? { label: "Normal (< 1.0 mm)", color: "emerald", icon: CheckCircle2 }
      : input.st_depression < 2.0
      ? { label: "Borderline (1.0-1.9 mm)", color: "amber", icon: AlertTriangle }
      : { label: "Significant Ischemia (≥ 2.0 mm)", color: "rose", icon: XCircle };

  return (
    <form
      onSubmit={onSubmit}
      id="patient-clinical-form"
      className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6"
    >
      {/* Header */}
      <div className="border-b border-slate-100 pb-4">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <Stethoscope className="w-5 h-5 text-indigo-600" />
            <h2 className="text-base font-bold text-slate-900">Enter Clinical Data Values</h2>
          </div>
          <span className="text-[11px] font-bold text-indigo-700 bg-indigo-50 px-2.5 py-1 rounded-full border border-indigo-200">
            Clinical Values Input
          </span>
        </div>
        <p className="text-xs text-slate-600">
          Enter the patient demographics (<strong>Age</strong>, <strong>Sex</strong>) and cardiovascular indicators (<strong>Serum Cholesterol</strong>, <strong>Resting Blood Pressure</strong>, <strong>Angina</strong>, and <strong>ST Segment Depression</strong>) below. Click <strong>Run Prediction</strong> to evaluate accurate CAD Positive / CAD Negative diagnosis.
        </p>

        {/* Quick Fill Sample Values Buttons */}
        <div className="mt-4 pt-3 border-t border-slate-100 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold text-slate-600 uppercase tracking-wider">
              Quick Fill Sample Values
            </span>
            <span className="text-[10px] text-slate-400">Click to autofill or type custom values below</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {PRESETS.map((p, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onChange(p.values)}
                className="text-left p-2.5 rounded-lg border border-slate-200 hover:border-indigo-400 hover:bg-indigo-50/50 transition-all cursor-pointer group"
              >
                <div className="text-xs font-bold text-slate-800 group-hover:text-indigo-900">
                  {p.label}
                </div>
                <div className="text-[10px] text-slate-500 truncate">{p.desc}</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Clinical Form Value Inputs */}
      <div className="space-y-5">
        {/* Row 1: Demographics (Age & Sex) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Patient Age */}
          <div className="p-4 bg-slate-50/80 rounded-xl border border-slate-200 space-y-2.5">
            <div className="flex items-center justify-between">
              <label htmlFor="input-patient-age" className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-[11px] font-bold flex items-center justify-center">1</span>
                <span>Patient Age</span>
              </label>
              <span
                className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                  ageStatus.color === "emerald"
                    ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                    : ageStatus.color === "amber"
                    ? "bg-amber-50 text-amber-800 border-amber-300"
                    : "bg-rose-50 text-rose-800 border-rose-300"
                }`}
              >
                {ageStatus.label}
              </span>
            </div>

            <div className="relative">
              <input
                id="input-patient-age"
                name="age"
                type="number"
                min="18"
                max="100"
                step="1"
                required
                placeholder="Enter age in years (e.g. 58)"
                value={input.age === 0 ? "" : input.age}
                onChange={(e) => onChange({ age: e.target.value === "" ? 0 : Number(e.target.value) })}
                className="w-full text-base font-mono font-bold text-slate-900 bg-white border-2 border-slate-300 focus:border-indigo-600 focus:ring-4 focus:ring-indigo-100 rounded-xl px-3.5 py-2.5 outline-none transition-all placeholder:text-slate-400 placeholder:font-normal pr-20 shadow-xs"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-bold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200 pointer-events-none">
                Years
              </span>
            </div>

            <div className="text-[11px] text-slate-500 pt-0.5">
              <span>Cardiovascular risk increases with age (&gt; 45 in men, &gt; 55 in women)</span>
            </div>
          </div>

          {/* Biological Sex */}
          <div className="p-4 bg-slate-50/80 rounded-xl border border-slate-200 space-y-2.5">
            <div className="flex items-center justify-between">
              <label htmlFor="input-patient-sex" className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-[11px] font-bold flex items-center justify-center">2</span>
                <span>Biological Sex</span>
              </label>
              <span className="text-[11px] font-bold px-2.5 py-0.5 rounded-full border bg-indigo-50 text-indigo-800 border-indigo-200">
                {sexStatus.label}
              </span>
            </div>

            {/* Direct selector buttons for Male / Female */}
            <div className="grid grid-cols-2 gap-2">
              <button
                id="input-sex-male"
                type="button"
                onClick={() => onChange({ sex: 1 })}
                className={`p-2.5 rounded-xl border-2 text-left transition-all cursor-pointer ${
                  input.sex === 1
                    ? "border-indigo-600 bg-indigo-50 text-indigo-950 ring-2 ring-indigo-500/20 font-bold shadow-xs"
                    : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1">
                    <User className="w-3.5 h-3.5" /> Male (1)
                  </span>
                  {input.sex === 1 && (
                    <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-indigo-100 text-indigo-800">
                      Selected
                    </span>
                  )}
                </div>
                <p className="text-[10px] text-slate-500 mt-0.5">Higher historical early CAD incidence</p>
              </button>

              <button
                id="input-sex-female"
                type="button"
                onClick={() => onChange({ sex: 0 })}
                className={`p-2.5 rounded-xl border-2 text-left transition-all cursor-pointer ${
                  input.sex === 0
                    ? "border-purple-600 bg-purple-50 text-purple-950 ring-2 ring-purple-500/20 font-bold shadow-xs"
                    : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1">
                    <UserCheck className="w-3.5 h-3.5" /> Female (0)
                  </span>
                  {input.sex === 0 && (
                    <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-purple-100 text-purple-800">
                      Selected
                    </span>
                  )}
                </div>
                <p className="text-[10px] text-slate-500 mt-0.5">Cardiovascular profile (estrogenic protection)</p>
              </button>
            </div>

            <div className="text-[11px] text-slate-500 pt-0.5">
              <span>Encoding: 1 = Male | 0 = Female (Cleveland Study standard)</span>
            </div>
          </div>
        </div>

        {/* 3. Serum Cholesterol */}
        <div className="p-4 bg-slate-50/80 rounded-xl border border-slate-200 space-y-2.5">
          <div className="flex items-center justify-between">
            <label htmlFor="input-serum-cholesterol" className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-[11px] font-bold flex items-center justify-center">3</span>
              <span>Serum Cholesterol Value</span>
            </label>
            <span
              className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                cholStatus.color === "emerald"
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                  : cholStatus.color === "amber"
                  ? "bg-amber-50 text-amber-800 border-amber-300"
                  : "bg-rose-50 text-rose-800 border-rose-300"
              }`}
            >
              {cholStatus.label}
            </span>
          </div>

          <div className="relative">
            <input
              id="input-serum-cholesterol"
              name="cholesterol"
              type="number"
              min="50"
              max="600"
              step="1"
              required
              placeholder="Enter cholesterol value (e.g. 215)"
              value={input.cholesterol === 0 ? "" : input.cholesterol}
              onChange={(e) => onChange({ cholesterol: e.target.value === "" ? 0 : Number(e.target.value) })}
              className="w-full text-base font-mono font-bold text-slate-900 bg-white border-2 border-slate-300 focus:border-indigo-600 focus:ring-4 focus:ring-indigo-100 rounded-xl px-3.5 py-2.5 outline-none transition-all placeholder:text-slate-400 placeholder:font-normal pr-20 shadow-xs"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-bold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200 pointer-events-none">
              mg/dL
            </span>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500 pt-0.5">
            <span>Reference: Desirable &lt; 200 mg/dL | Borderline 200–239 mg/dL | High Risk ≥ 240 mg/dL</span>
          </div>
        </div>

        {/* 4. Resting Blood Pressure */}
        <div className="p-4 bg-slate-50/80 rounded-xl border border-slate-200 space-y-2.5">
          <div className="flex items-center justify-between">
            <label htmlFor="input-resting-bp" className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-[11px] font-bold flex items-center justify-center">4</span>
              <span>Resting Blood Pressure Value</span>
            </label>
            <span
              className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                bpStatus.color === "emerald"
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                  : bpStatus.color === "amber"
                  ? "bg-amber-50 text-amber-800 border-amber-300"
                  : "bg-rose-50 text-rose-800 border-rose-300"
              }`}
            >
              {bpStatus.label}
            </span>
          </div>

          <div className="relative">
            <input
              id="input-resting-bp"
              name="resting_bp"
              type="number"
              min="50"
              max="300"
              step="1"
              required
              placeholder="Enter blood pressure value (e.g. 125)"
              value={input.resting_bp === 0 ? "" : input.resting_bp}
              onChange={(e) => onChange({ resting_bp: e.target.value === "" ? 0 : Number(e.target.value) })}
              className="w-full text-base font-mono font-bold text-slate-900 bg-white border-2 border-slate-300 focus:border-indigo-600 focus:ring-4 focus:ring-indigo-100 rounded-xl px-3.5 py-2.5 outline-none transition-all placeholder:text-slate-400 placeholder:font-normal pr-20 shadow-xs"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-bold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200 pointer-events-none">
              mm Hg
            </span>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500 pt-0.5">
            <span>Reference: Normal &lt; 120 mm Hg | Elevated 120–129 mm Hg | Hypertensive ≥ 130 mm Hg</span>
          </div>
        </div>

        {/* 5. Exercise-Induced Angina */}
        <div className="p-4 bg-slate-50/80 rounded-xl border border-slate-200 space-y-2.5">
          <div className="flex items-center justify-between">
            <label htmlFor="input-angina-value" className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-[11px] font-bold flex items-center justify-center">5</span>
              <span>Angina Value (Exercise-Induced Chest Pain)</span>
            </label>
            <span
              className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                anginaStatus.color === "emerald"
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                  : "bg-rose-50 text-rose-800 border-rose-300"
              }`}
            >
              {anginaStatus.label}
            </span>
          </div>

          {/* Option to type numeric value 0 or 1 */}
          <div className="flex gap-2.5">
            <div className="relative flex-1">
              <input
                id="input-angina-value"
                name="exercise_angina"
                type="number"
                min="0"
                max="1"
                step="1"
                required
                placeholder="Enter 0 or 1"
                value={input.exercise_angina}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  onChange({ exercise_angina: val === 1 ? 1 : 0 });
                }}
                className="w-full text-base font-mono font-bold text-slate-900 bg-white border-2 border-slate-300 focus:border-indigo-600 focus:ring-4 focus:ring-indigo-100 rounded-xl px-3.5 py-2.5 outline-none transition-all pr-24 shadow-xs"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-bold text-slate-500 bg-slate-100 px-2 py-1 rounded-md border border-slate-200 pointer-events-none">
                Value: {input.exercise_angina}
              </span>
            </div>
          </div>

          {/* Direct selection buttons for value 0 or value 1 */}
          <div className="grid grid-cols-2 gap-2.5 pt-1">
            <button
              id="input-angina-no"
              type="button"
              onClick={() => onChange({ exercise_angina: 0 })}
              className={`p-3 rounded-xl border-2 text-left transition-all cursor-pointer ${
                input.exercise_angina === 0
                  ? "border-emerald-500 bg-emerald-50 text-emerald-950 ring-2 ring-emerald-500/20 font-bold shadow-xs"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
              }`}
            >
              <div className="flex items-center justify-between text-xs">
                <span>Value 0: No Angina</span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
                  Normal
                </span>
              </div>
              <p className="text-[10px] text-slate-500 mt-1">No chest pain or tightness during physical exertion</p>
            </button>

            <button
              id="input-angina-yes"
              type="button"
              onClick={() => onChange({ exercise_angina: 1 })}
              className={`p-3 rounded-xl border-2 text-left transition-all cursor-pointer ${
                input.exercise_angina === 1
                  ? "border-rose-500 bg-rose-50 text-rose-950 ring-2 ring-rose-500/20 font-bold shadow-xs"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
              }`}
            >
              <div className="flex items-center justify-between text-xs">
                <span className="text-rose-900 font-bold">Value 1: Angina Present</span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-rose-100 text-rose-800">
                  CAD Risk
                </span>
              </div>
              <p className="text-[10px] text-rose-700/80 mt-1">Ischemic chest discomfort or pain triggered by exertion</p>
            </button>
          </div>

          <div className="text-[11px] text-slate-500 pt-0.5">
            <span>Reference: 0 = Absent (Normal) | 1 = Present (CAD Ischemia Indicator)</span>
          </div>
        </div>

        {/* 6. ST-Segment Depression (Oldpeak) */}
        <div className="p-4 bg-slate-50/80 rounded-xl border border-slate-200 space-y-2.5">
          <div className="flex items-center justify-between">
            <label htmlFor="input-st-depression" className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
              <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-[11px] font-bold flex items-center justify-center">6</span>
              <span>ST Segment Depression Value (Oldpeak)</span>
            </label>
            <span
              className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                stStatus.color === "emerald"
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                  : stStatus.color === "amber"
                  ? "bg-amber-50 text-amber-800 border-amber-300"
                  : "bg-rose-50 text-rose-800 border-rose-300"
              }`}
            >
              {stStatus.label}
            </span>
          </div>

          <div className="relative">
            <input
              id="input-st-depression"
              name="st_depression"
              type="number"
              min="0.0"
              max="10.0"
              step="0.1"
              required
              placeholder="Enter ST depression in mm (e.g. 1.2)"
              value={isNaN(input.st_depression) ? "" : input.st_depression}
              onChange={(e) => onChange({ st_depression: e.target.value === "" ? 0 : Number(e.target.value) })}
              className="w-full text-base font-mono font-bold text-slate-900 bg-white border-2 border-slate-300 focus:border-indigo-600 focus:ring-4 focus:ring-indigo-100 rounded-xl px-3.5 py-2.5 outline-none transition-all placeholder:text-slate-400 placeholder:font-normal pr-20 shadow-xs"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-bold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200 pointer-events-none">
              mm
            </span>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500 pt-0.5">
            <span>Reference: Normal &lt; 1.0 mm | Borderline 1.0–1.9 mm | Significant Ischemia ≥ 2.0 mm</span>
          </div>
        </div>

        {/* Quantum Feature Map Architecture */}
        <div className="pt-2 border-t border-slate-100">
          <label className="block text-xs font-bold text-slate-700 mb-1.5">
            Quantum Hilbert Feature Map Architecture
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => onChange({ feature_map: "zz_feature_map" })}
              className={`p-2.5 rounded-lg border text-left text-xs transition-all cursor-pointer ${
                input.feature_map === "zz_feature_map"
                  ? "border-indigo-600 bg-indigo-50/80 text-indigo-950 ring-2 ring-indigo-500/20 font-semibold"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
              }`}
            >
              <div className="font-bold flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                ZZ-Feature Map
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                CNOT entangled non-linear Hilbert space
              </div>
            </button>

            <button
              type="button"
              onClick={() => onChange({ feature_map: "angle" })}
              className={`p-2.5 rounded-lg border text-left text-xs transition-all cursor-pointer ${
                input.feature_map === "angle"
                  ? "border-indigo-600 bg-indigo-50/80 text-indigo-950 ring-2 ring-indigo-500/20 font-semibold"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
              }`}
            >
              <div className="font-bold flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-indigo-600" />
                Angle Embedding
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                Separable Ry single-qubit rotations
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Summary of Entered Values Ready to Evaluate */}
      <div className="p-3.5 bg-indigo-50/50 rounded-xl border border-indigo-100 text-xs">
        <div className="font-bold text-indigo-950 mb-1 flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-indigo-600" />
          <span>Entered Clinical Values for CAD Evaluation:</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 text-[11px] font-mono mt-2">
          <div className="bg-white p-2 rounded border border-indigo-100">
            <span className="text-slate-400 block text-[10px]">Age:</span>
            <span className="font-bold text-slate-900">{input.age} yrs</span>
          </div>
          <div className="bg-white p-2 rounded border border-indigo-100">
            <span className="text-slate-400 block text-[10px]">Sex:</span>
            <span className="font-bold text-slate-900">{input.sex === 1 ? "Male (1)" : "Female (0)"}</span>
          </div>
          <div className="bg-white p-2 rounded border border-indigo-100">
            <span className="text-slate-400 block text-[10px]">Cholesterol:</span>
            <span className="font-bold text-slate-900">{input.cholesterol} mg/dL</span>
          </div>
          <div className="bg-white p-2 rounded border border-indigo-100">
            <span className="text-slate-400 block text-[10px]">Blood Pressure:</span>
            <span className="font-bold text-slate-900">{input.resting_bp} mm Hg</span>
          </div>
          <div className="bg-white p-2 rounded border border-indigo-100">
            <span className="text-slate-400 block text-[10px]">Angina:</span>
            <span className="font-bold text-slate-900">Value {input.exercise_angina} ({input.exercise_angina === 1 ? "Yes" : "No"})</span>
          </div>
          <div className="bg-white p-2 rounded border border-indigo-100">
            <span className="text-slate-400 block text-[10px]">ST Depression:</span>
            <span className="font-bold text-slate-900">{input.st_depression.toFixed(1)} mm</span>
          </div>
        </div>
      </div>

      {/* Primary Dedicated RUN Button */}
      <button
        id="run-cad-button"
        type="submit"
        disabled={loading}
        className="w-full py-4 px-6 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white rounded-xl shadow-md hover:shadow-indigo-500/25 transition-all cursor-pointer flex items-center justify-center gap-3 disabled:opacity-50 group border border-indigo-500"
      >
        {loading ? (
          <>
            <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
            <span className="text-sm font-bold tracking-wide">COMPUTING QUANTUM CAD PREDICTION...</span>
          </>
        ) : (
          <>
            <div className="w-9 h-9 rounded-lg bg-white/20 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Play className="w-5 h-5 fill-white text-white translate-x-0.5" />
            </div>
            <div className="text-left">
              <div className="text-base font-black tracking-wide leading-tight uppercase">
                RUN PREDICTION
              </div>
              <div className="text-[11px] text-indigo-100 font-medium leading-tight">
                Evaluate Entered Values for CAD Positive / CAD Negative
              </div>
            </div>
          </>
        )}
      </button>
    </form>
  );
};
