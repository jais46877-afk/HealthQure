import React, { useState, useRef } from "react";
import { BatchPatientRecord, ClinicalInput } from "../types";
import {
  Upload,
  FileSpreadsheet,
  Download,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  ArrowRight,
  RefreshCw,
  Cpu,
  FileText,
  HelpCircle,
  Eye
} from "lucide-react";

interface ClinicalDataUploaderProps {
  onSelectPatientForCircuit: (patient: ClinicalInput & { name?: string; id?: string }) => void;
  activeFeatureMap: "zz_feature_map" | "angle";
}

export const ClinicalDataUploader: React.FC<ClinicalDataUploaderProps> = ({
  onSelectPatientForCircuit,
  activeFeatureMap
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [parsedPatients, setParsedPatients] = useState<BatchPatientRecord[]>([]);
  const [batchResults, setBatchResults] = useState<BatchPatientRecord[] | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sample CSV content generator
  const SAMPLE_CSV_CONTENT = `patient_id,patient_name,cholesterol,resting_bp,exercise_angina,st_depression
PT-01,Johnathan Davis,285,152,1,2.8
PT-02,Eleanor Vance,175,114,0,0.2
PT-03,Marcus Sterling,240,138,1,1.9
PT-04,Clara Henderson,188,118,0,0.1
PT-05,Arthur Pendelton,310,165,1,3.6
PT-06,Sophia Martinez,210,124,0,0.5`;

  const downloadSampleCSV = () => {
    const blob = new Blob([SAMPLE_CSV_CONTENT], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "cad_clinical_sample_data.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const loadDemoCohort = () => {
    parseCSVContent(SAMPLE_CSV_CONTENT, "demo_cad_cohort.csv");
  };

  const parseCSVContent = (text: string, name: string) => {
    setErrorMessage(null);
    setBatchResults(null);

    const lines = text.trim().split(/\r\n|\n/);
    if (lines.length < 2) {
      setErrorMessage("File must contain a header row and at least one clinical patient record.");
      return;
    }

    const headers = lines[0].split(",").map((h) => h.trim().toLowerCase().replace(/['"]+/g, ""));

    // Find column index matches flexibly
    const cholIdx = headers.findIndex((h) =>
      h.includes("chol") || h.includes("serum")
    );
    const bpIdx = headers.findIndex((h) =>
      h.includes("bp") || h.includes("blood") || h.includes("trestbps") || h.includes("press")
    );
    const anginaIdx = headers.findIndex((h) =>
      h.includes("angina") || h.includes("exang")
    );
    const stIdx = headers.findIndex((h) =>
      h.includes("st") || h.includes("depression") || h.includes("oldpeak")
    );
    const idIdx = headers.findIndex((h) =>
      h.includes("id") || h.includes("patient_id")
    );
    const nameIdx = headers.findIndex((h) =>
      h.includes("name")
    );

    if (cholIdx === -1 || bpIdx === -1 || anginaIdx === -1 || stIdx === -1) {
      setErrorMessage(
        "Could not detect all 4 mandatory clinical parameters: Serum Cholesterol, Resting BP, Angina, ST Depression. Please check your headers."
      );
      return;
    }

    const patients: BatchPatientRecord[] = [];

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      const cols = line.split(",").map((c) => c.trim().replace(/['"]+/g, ""));
      const chol = parseFloat(cols[cholIdx]);
      const bp = parseFloat(cols[bpIdx]);

      // Handle angina formatting (1, 0, yes, no, true, false)
      const rawAngina = cols[anginaIdx]?.toLowerCase();
      let angina = 0;
      if (rawAngina === "1" || rawAngina === "yes" || rawAngina === "true" || rawAngina === "positive") {
        angina = 1;
      } else {
        angina = 0;
      }

      const st = parseFloat(cols[stIdx]);
      const patientId = idIdx !== -1 && cols[idIdx] ? cols[idIdx] : `PT-${i.toString().padStart(3, "0")}`;
      const patientName = nameIdx !== -1 && cols[nameIdx] ? cols[nameIdx] : `Patient #${i}`;

      if (isNaN(chol) || isNaN(bp) || isNaN(st)) {
        continue;
      }

      patients.push({
        id: patientId,
        name: patientName,
        cholesterol: chol,
        resting_bp: bp,
        exercise_angina: angina,
        st_depression: st
      });
    }

    if (patients.length === 0) {
      setErrorMessage("No valid numeric patient records could be extracted from the file.");
      return;
    }

    setFileName(name);
    setParsedPatients(patients);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      if (file.name.endsWith(".json")) {
        try {
          const parsed = JSON.parse(content);
          const list = Array.isArray(parsed) ? parsed : [parsed];
          const formatted: BatchPatientRecord[] = list.map((item, idx) => ({
            id: item.id || `PT-${(idx + 1).toString().padStart(3, "0")}`,
            name: item.name || `Patient #${idx + 1}`,
            cholesterol: Number(item.cholesterol || item.serum_cholesterol || 200),
            resting_bp: Number(item.resting_bp || item.blood_pressure || item.bp || 120),
            exercise_angina: item.exercise_angina === 1 || item.exercise_angina === true || item.angina === 1 || item.angina === true ? 1 : 0,
            st_depression: Number(item.st_depression || item.oldpeak || 0.0)
          }));
          setFileName(file.name);
          setParsedPatients(formatted);
          setBatchResults(null);
          setErrorMessage(null);
        } catch (err) {
          setErrorMessage("Failed to parse JSON file. Ensure it contains an array of patient records.");
        }
      } else {
        parseCSVContent(content, file.name);
      }
    };
    reader.readAsText(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        if (file.name.endsWith(".json")) {
          try {
            const parsed = JSON.parse(content);
            const list = Array.isArray(parsed) ? parsed : [parsed];
            const formatted: BatchPatientRecord[] = list.map((item, idx) => ({
              id: item.id || `PT-${(idx + 1).toString().padStart(3, "0")}`,
              name: item.name || `Patient #${idx + 1}`,
              cholesterol: Number(item.cholesterol || 200),
              resting_bp: Number(item.resting_bp || 120),
              exercise_angina: item.exercise_angina ? 1 : 0,
              st_depression: Number(item.st_depression || 0)
            }));
            setFileName(file.name);
            setParsedPatients(formatted);
            setBatchResults(null);
            setErrorMessage(null);
          } catch {
            setErrorMessage("Failed to parse JSON file.");
          }
        } else {
          parseCSVContent(content, file.name);
        }
      };
      reader.readAsText(file);
    }
  };

  // Run batch classification with PennyLane QSVC
  const runBatchPrediction = async () => {
    if (parsedPatients.length === 0) return;
    setIsProcessing(true);
    setErrorMessage(null);

    try {
      const response = await fetch("/api/batch-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patients: parsedPatients })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.details || err.error || "Batch prediction failed");
      }

      const data = await response.json();
      setBatchResults(data.results);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to complete quantum batch prediction.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Export results to CSV
  const exportResultsCSV = () => {
    if (!batchResults) return;

    const headers = [
      "patient_id",
      "patient_name",
      "cholesterol",
      "resting_bp",
      "exercise_angina",
      "st_depression",
      "cad_diagnosis",
      "cad_status",
      "risk_level",
      "chd_probability_percent",
      "decision_margin"
    ];

    const rows = batchResults.map((p) => [
      p.id,
      `"${p.name || ""}"`,
      p.cholesterol,
      p.resting_bp,
      p.exercise_angina === 1 ? "Yes" : "No",
      p.st_depression,
      `"${p.cad_diagnosis || ""}"`,
      p.cad_status || "",
      p.risk_level || "",
      ((p.chd_probability || 0) * 100).toFixed(2),
      p.decision_margin?.toFixed(4) || ""
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `cad_quantum_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const positiveCount = batchResults ? batchResults.filter((r) => r.cad_status === "POSITIVE").length : 0;
  const negativeCount = batchResults ? batchResults.filter((r) => r.cad_status === "NEGATIVE").length : 0;

  return (
    <div id="clinical-uploader-panel" className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
      {/* Title & Description */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="w-5 h-5 text-indigo-600" />
            <h2 className="text-base font-bold text-slate-900">
              Upload Clinical Data for CAD Classification
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Upload CSV or JSON with <strong className="text-slate-700">Serum Cholesterol</strong>, <strong className="text-slate-700">Resting Blood Pressure</strong>, <strong className="text-slate-700">Angina</strong>, and <strong className="text-slate-700">ST Segment Depression</strong> to predict <strong className="text-indigo-700">CAD Positive</strong> or <strong className="text-indigo-700">CAD Negative</strong>.
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={downloadSampleCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg transition-colors cursor-pointer"
            title="Download formatted CSV template"
          >
            <Download className="w-3.5 h-3.5 text-slate-600" />
            <span>Sample CSV</span>
          </button>

          <button
            type="button"
            onClick={loadDemoCohort}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg transition-colors cursor-pointer"
            title="Populate with 6 test patient records"
          >
            <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
            <span>Load Demo Cohort</span>
          </button>
        </div>
      </div>

      {/* Error alert */}
      {errorMessage && (
        <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div>{errorMessage}</div>
        </div>
      )}

      {/* Drag and Drop Zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
          dragActive
            ? "border-indigo-500 bg-indigo-50/60"
            : "border-slate-200 hover:border-indigo-300 hover:bg-slate-50/50"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.json,text/csv,application/json"
          onChange={handleFileChange}
          className="hidden"
        />
        <div className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-600 mx-auto flex items-center justify-center mb-3">
          <Upload className="w-6 h-6" />
        </div>
        <div className="text-sm font-bold text-slate-800">
          {fileName ? (
            <span className="text-indigo-600 font-mono">{fileName}</span>
          ) : (
            "Drop your clinical CSV / JSON file here or click to browse"
          )}
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Supports headers: cholesterol, resting_bp, angina (0/1 or Yes/No), st_depression
        </p>
      </div>

      {/* Parsed Patients Summary & Action */}
      {parsedPatients.length > 0 && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 bg-slate-50 rounded-xl border border-slate-200">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center font-bold text-xs">
                {parsedPatients.length}
              </div>
              <div>
                <div className="text-xs font-bold text-slate-800">
                  {parsedPatients.length} Patient {parsedPatients.length === 1 ? "Record" : "Records"} Loaded
                </div>
                <div className="text-[11px] text-slate-500">
                  Ready for 4-qubit Hilbert space transformation & PennyLane QSVC classification
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={runBatchPrediction}
              disabled={isProcessing}
              className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white text-xs font-bold rounded-lg shadow-sm transition-all cursor-pointer disabled:opacity-50"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Quantum Kernels...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Predict CAD Positive / Negative</span>
                </>
              )}
            </button>
          </div>

          {/* Results Summary Dashboard (if batch executed) */}
          {batchResults && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                <div className="text-xs text-slate-500 font-semibold">Total Patients Evaluated</div>
                <div className="text-2xl font-bold font-mono text-slate-900 mt-1">
                  {batchResults.length}
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5">Quantum simulator: default.qubit</div>
              </div>

              <div className="p-4 bg-rose-50/70 rounded-xl border border-rose-200">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-rose-900 uppercase tracking-wider">CAD Positive</span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-100 text-rose-800">
                    High Risk
                  </span>
                </div>
                <div className="text-2xl font-bold font-mono text-rose-950 mt-1">
                  {positiveCount} <span className="text-xs font-normal text-rose-700">({((positiveCount / batchResults.length) * 100).toFixed(1)}%)</span>
                </div>
                <div className="text-[11px] text-rose-700 mt-0.5">Prompt cardiology referral indicated</div>
              </div>

              <div className="p-4 bg-emerald-50/70 rounded-xl border border-emerald-200">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-emerald-900 uppercase tracking-wider">CAD Negative</span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800">
                    Low Risk
                  </span>
                </div>
                <div className="text-2xl font-bold font-mono text-emerald-950 mt-1">
                  {negativeCount} <span className="text-xs font-normal text-emerald-700">({((negativeCount / batchResults.length) * 100).toFixed(1)}%)</span>
                </div>
                <div className="text-[11px] text-emerald-700 mt-0.5">No acute ischemic signatures</div>
              </div>
            </div>
          )}

          {/* Interactive Patients Table */}
          <div className="border border-slate-200 rounded-xl overflow-hidden shadow-xs">
            <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <span className="text-xs font-bold text-slate-700">
                {batchResults ? "Quantum Diagnostic Results & CAD Predictions" : "Uploaded Clinical Parameters Preview"}
              </span>
              {batchResults && (
                <button
                  type="button"
                  onClick={exportResultsCSV}
                  className="flex items-center gap-1 text-[11px] font-bold text-indigo-700 hover:text-indigo-900 cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export Predictions (CSV)</span>
                </button>
              )}
            </div>

            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="bg-slate-100/75 text-slate-600 sticky top-0 z-10 border-b border-slate-200 font-semibold">
                  <tr>
                    <th className="p-2.5">Patient ID</th>
                    <th className="p-2.5">Serum Cholesterol</th>
                    <th className="p-2.5">Resting BP</th>
                    <th className="p-2.5">Angina</th>
                    <th className="p-2.5">ST Depression</th>
                    {batchResults && (
                      <>
                        <th className="p-2.5">CAD Diagnosis</th>
                        <th className="p-2.5">Probability</th>
                        <th className="p-2.5">Action</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {(batchResults || parsedPatients).map((patient, idx) => {
                    const isResult = "cad_status" in patient;
                    const isPositive = patient.cad_status === "POSITIVE";

                    return (
                      <tr
                        key={idx}
                        className={`hover:bg-slate-50/80 transition-colors ${
                          isResult
                            ? isPositive
                              ? "bg-rose-50/20"
                              : "bg-emerald-50/20"
                            : ""
                        }`}
                      >
                        <td className="p-2.5 font-mono font-semibold text-slate-900">
                          {patient.id}
                          {patient.name && (
                            <span className="block text-[10px] font-sans font-normal text-slate-500">
                              {patient.name}
                            </span>
                          )}
                        </td>

                        <td className="p-2.5">
                          <span className="font-mono font-medium text-slate-800">{patient.cholesterol}</span>
                          <span className="text-[10px] text-slate-400 ml-1">mg/dL</span>
                        </td>

                        <td className="p-2.5">
                          <span className="font-mono font-medium text-slate-800">{patient.resting_bp}</span>
                          <span className="text-[10px] text-slate-400 ml-1">mmHg</span>
                        </td>

                        <td className="p-2.5">
                          {patient.exercise_angina === 1 ? (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-900 border border-amber-300">
                              Yes (Induced)
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600">
                              No
                            </span>
                          )}
                        </td>

                        <td className="p-2.5 font-mono text-slate-800">
                          {patient.st_depression.toFixed(1)} mm
                        </td>

                        {isResult && (
                          <>
                            <td className="p-2.5">
                              {isPositive ? (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-extrabold bg-rose-600 text-white shadow-xs">
                                  <AlertCircle className="w-3 h-3" />
                                  CAD Positive
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-extrabold bg-emerald-600 text-white shadow-xs">
                                  <CheckCircle2 className="w-3 h-3" />
                                  CAD Negative
                                </span>
                              )}
                            </td>

                            <td className="p-2.5 font-mono font-bold text-slate-900">
                              {((patient.chd_probability || 0) * 100).toFixed(1)}%
                              <span className="block text-[10px] font-normal text-slate-400">
                                f(x): {patient.decision_margin?.toFixed(2)}
                              </span>
                            </td>

                            <td className="p-2.5">
                              <button
                                type="button"
                                onClick={() =>
                                  onSelectPatientForCircuit({
                                    cholesterol: patient.cholesterol,
                                    resting_bp: patient.resting_bp,
                                    exercise_angina: patient.exercise_angina,
                                    st_depression: patient.st_depression,
                                    feature_map: activeFeatureMap,
                                    id: patient.id,
                                    name: patient.name
                                  })
                                }
                                className="flex items-center gap-1 text-[11px] font-bold text-indigo-600 hover:text-indigo-800 hover:underline cursor-pointer"
                                title="Load this patient into the 4-Qubit Circuit & Bloch Sphere view"
                              >
                                <Eye className="w-3.5 h-3.5" />
                                <span>Inspect Circuit</span>
                              </button>
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
