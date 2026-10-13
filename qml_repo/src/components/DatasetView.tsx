import React, { useState } from "react";
import { DatasetResponse } from "../types";
import { Database, Search, Filter, HeartPulse } from "lucide-react";

interface DatasetViewProps {
  dataset?: DatasetResponse;
}

export const DatasetView: React.FC<DatasetViewProps> = ({ dataset }) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterRisk, setFilterRisk] = useState<string>("all");

  const records = dataset?.sample_records || [];
  const total = dataset?.total_records || records.length;
  const pos = dataset?.positive_count || records.filter((r) => r.chd_risk === 1).length;
  const neg = dataset?.negative_count || records.filter((r) => r.chd_risk === 0).length;

  const filtered = records.filter((r) => {
    if (filterRisk === "high" && r.chd_risk !== 1) return false;
    if (filterRisk === "low" && r.chd_risk !== 0) return false;
    if (searchTerm) {
      const s = searchTerm.toLowerCase();
      return (
        r.cholesterol.toString().includes(s) ||
        r.resting_bp.toString().includes(s) ||
        r.st_depression.toString().includes(s)
      );
    }
    return true;
  });

  return (
    <div id="clinical-dataset-view" className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-indigo-600" />
            <h3 className="text-base font-semibold text-slate-900">
              Coronary Heart Disease Clinical Dataset
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Standardized cohort based on Cleveland & Framingham Heart Study cardiovascular indicators
          </p>
        </div>

        {/* Cohort Stats */}
        <div className="flex items-center gap-3 text-xs">
          <div className="px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-slate-500">Total Cohort: </span>
            <span className="font-bold font-mono text-slate-800">{total}</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800">
            <span>CAD Positive: </span>
            <span className="font-bold font-mono">{pos}</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800">
            <span>CAD Negative: </span>
            <span className="font-bold font-mono">{neg}</span>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search by cholesterol, BP, or ST depression..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-xs border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>

        <div className="flex items-center gap-2 text-xs">
          <Filter className="w-4 h-4 text-slate-400" />
          <span className="text-slate-500">Filter Risk:</span>
          <select
            value={filterRisk}
            onChange={(e) => setFilterRisk(e.target.value)}
            className="border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="all">All Patients</option>
            <option value="high">CAD Positive (High Risk)</option>
            <option value="low">CAD Negative (Low Risk)</option>
          </select>
        </div>
      </div>

      {/* Clinical records table */}
      <div className="overflow-x-auto border border-slate-200 rounded-lg">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
            <tr>
              <th className="py-2.5 px-3">Patient ID</th>
              <th className="py-2.5 px-3">Age</th>
              <th className="py-2.5 px-3">Sex</th>
              <th className="py-2.5 px-3">Cholesterol (mg/dL)</th>
              <th className="py-2.5 px-3">Resting BP (mm Hg)</th>
              <th className="py-2.5 px-3">Exercise Angina</th>
              <th className="py-2.5 px-3">ST Depression (Oldpeak)</th>
              <th className="py-2.5 px-3">Clinical Diagnosis</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {filtered.slice(0, 15).map((record, idx) => (
              <tr key={idx} className="hover:bg-slate-50/70 transition-colors">
                <td className="py-2.5 px-3 font-mono text-slate-500">#{(idx + 1).toString().padStart(3, "0")}</td>
                <td className="py-2.5 px-3 font-mono text-slate-800">
                  {record.age ? Math.round(record.age) : "—"} <span className="text-[10px] text-slate-400">yr</span>
                </td>
                <td className="py-2.5 px-3 text-slate-800 font-medium">
                  {record.sex === 1 ? (
                    <span className="text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded text-[11px]">M (1)</span>
                  ) : record.sex === 0 ? (
                    <span className="text-purple-700 bg-purple-50 px-1.5 py-0.5 rounded text-[11px]">F (0)</span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="py-2.5 px-3 font-mono font-medium text-slate-800">
                  {record.cholesterol}
                  <span className="text-[10px] text-slate-400 ml-1">mg/dL</span>
                </td>
                <td className="py-2.5 px-3 font-mono text-slate-800">
                  {record.resting_bp}
                  <span className="text-[10px] text-slate-400 ml-1">mm Hg</span>
                </td>
                <td className="py-2.5 px-3">
                  {record.exercise_angina === 1 ? (
                    <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-rose-700 bg-rose-50 px-2 py-0.5 rounded">
                      <HeartPulse className="w-3 h-3" /> Yes
                    </span>
                  ) : (
                    <span className="text-[11px] text-slate-600 bg-slate-100 px-2 py-0.5 rounded">
                      No
                    </span>
                  )}
                </td>
                <td className="py-2.5 px-3 font-mono text-slate-800">
                  {record.st_depression.toFixed(2)} mm
                </td>
                <td className="py-2.5 px-3">
                  {record.chd_risk === 1 ? (
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-rose-100 text-rose-800 border border-rose-200">
                      CAD Positive
                    </span>
                  ) : (
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                      CAD Negative
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="text-[11px] text-slate-400 flex items-center justify-between">
        <span>Showing {Math.min(15, filtered.length)} of {filtered.length} matching cohort records</span>
        <span>Standardized via scikit-learn StandardScaler + PCA</span>
      </div>
    </div>
  );
};
