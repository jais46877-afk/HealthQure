import React from "react";
import { ModelMetrics } from "../types";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine
} from "recharts";
import { Award, ShieldCheck, Zap, Layers } from "lucide-react";

interface MetricsViewProps {
  metrics: ModelMetrics;
  onRetrain?: () => void;
  isTraining?: boolean;
}

export const MetricsView: React.FC<MetricsViewProps> = ({
  metrics,
  onRetrain,
  isTraining = false
}) => {
  // Confusion matrix breakdown
  const cm = metrics.confusion_matrix || [[0, 0], [0, 0]];
  const tn = cm[0]?.[0] ?? 0;
  const fp = cm[0]?.[1] ?? 0;
  const fn = cm[1]?.[0] ?? 0;
  const tp = cm[1]?.[1] ?? 0;

  const precision = tp + fp > 0 ? (tp / (tp + fp)).toFixed(3) : "0.000";
  const recall = tp + fn > 0 ? (tp / (tp + fn)).toFixed(3) : "0.000";
  const specificity = tn + fp > 0 ? (tn / (tn + fp)).toFixed(3) : "0.000";

  // Format ROC curve data for recharts
  const rocData = (metrics.roc_curve || []).map((pt) => ({
    fpr: pt.fpr,
    tpr: pt.tpr,
    baseline: pt.fpr
  }));

  return (
    <div id="quantum-metrics-view" className="space-y-6">
      {/* Top Metric Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium mb-1">
            <span>ROC-AUC SCORE</span>
            <Award className="w-4 h-4 text-indigo-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900 font-mono">
            {metrics.roc_auc.toFixed(3)}
          </div>
          <p className="text-[11px] text-emerald-600 font-medium mt-1">High discrimination power</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium mb-1">
            <span>TEST ACCURACY</span>
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900 font-mono">
            {(metrics.accuracy * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Stratified test split</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium mb-1">
            <span>F1-SCORE</span>
            <Zap className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-2xl font-bold text-slate-900 font-mono">
            {metrics.f1_score.toFixed(3)}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Harmonic precision/recall</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between text-xs text-slate-500 font-medium mb-1">
            <span>SUPPORT VECTORS</span>
            <Layers className="w-4 h-4 text-purple-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900 font-mono">
            {metrics.n_support_vectors}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Quantum boundary states</p>
        </div>
      </div>

      {/* ROC-AUC Chart & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROC-AUC Curve */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-900">ROC Evaluation Curve</h3>
              <p className="text-xs text-slate-500">True Positive Rate vs. False Positive Rate</p>
            </div>
            <span className="text-xs font-mono font-medium px-2.5 py-1 bg-indigo-50 border border-indigo-200 text-indigo-700 rounded-md">
              AUC = {metrics.roc_auc.toFixed(3)}
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="fpr"
                  type="number"
                  domain={[0, 1]}
                  tickCount={6}
                  label={{ value: "False Positive Rate (1 - Specificity)", position: "insideBottom", offset: -5, fontSize: 10, fill: "#64748b" }}
                  tick={{ fontSize: 10, fill: "#64748b" }}
                />
                <YAxis
                  dataKey="tpr"
                  type="number"
                  domain={[0, 1]}
                  tickCount={6}
                  label={{ value: "True Positive Rate (Sensitivity)", angle: -90, position: "insideLeft", offset: 15, fontSize: 10, fill: "#64748b" }}
                  tick={{ fontSize: 10, fill: "#64748b" }}
                />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload || !payload.length) return null;
                    const data = payload[0].payload;
                    return (
                      <div className="bg-slate-900 text-white text-xs p-2 rounded shadow font-mono">
                        <div>FPR: {data.fpr.toFixed(3)}</div>
                        <div>TPR: {data.tpr.toFixed(3)}</div>
                      </div>
                    );
                  }}
                />
                <ReferenceLine stroke="#94a3b8" strokeDasharray="4 4" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
                <Line
                  type="monotone"
                  dataKey="tpr"
                  stroke="#4f46e5"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#4f46e5" }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix & Diagnostics */}
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Clinical Confusion Matrix</h3>
            <p className="text-xs text-slate-500 mb-4">Predicted vs. Actual Coronary Heart Disease classes</p>

            <div className="grid grid-cols-2 gap-3 mb-5">
              <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-center">
                <div className="text-xs text-emerald-800 font-medium">True Negative (TN)</div>
                <div className="text-2xl font-bold text-emerald-950 font-mono mt-1">{tn}</div>
                <div className="text-[10px] text-emerald-600">Correctly identified low risk</div>
              </div>

              <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-center">
                <div className="text-xs text-amber-800 font-medium">False Positive (FP)</div>
                <div className="text-2xl font-bold text-amber-950 font-mono mt-1">{fp}</div>
                <div className="text-[10px] text-amber-600">Type I error (over-flagged)</div>
              </div>

              <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-center">
                <div className="text-xs text-rose-800 font-medium">False Negative (FN)</div>
                <div className="text-2xl font-bold text-rose-950 font-mono mt-1">{fn}</div>
                <div className="text-[10px] text-rose-600">Type II error (missed ischemia)</div>
              </div>

              <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg text-center">
                <div className="text-xs text-indigo-800 font-medium">True Positive (TP)</div>
                <div className="text-2xl font-bold text-indigo-950 font-mono mt-1">{tp}</div>
                <div className="text-[10px] text-indigo-600">Confirmed CHD ischemic risk</div>
              </div>
            </div>

            {/* Diagnostic Ratios */}
            <div className="grid grid-cols-3 gap-2 text-center text-xs border-t border-slate-100 pt-3">
              <div>
                <div className="text-slate-500">Precision (PPV)</div>
                <div className="font-mono font-bold text-slate-900 mt-0.5">{precision}</div>
              </div>
              <div>
                <div className="text-slate-500">Sensitivity (Recall)</div>
                <div className="font-mono font-bold text-slate-900 mt-0.5">{recall}</div>
              </div>
              <div>
                <div className="text-slate-500">Specificity</div>
                <div className="font-mono font-bold text-slate-900 mt-0.5">{specificity}</div>
              </div>
            </div>
          </div>

          {onRetrain && (
            <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between">
              <span className="text-xs text-slate-500">Need fresh optimization?</span>
              <button
                onClick={onRetrain}
                disabled={isTraining}
                className="px-4 py-2 text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg transition-colors cursor-pointer disabled:opacity-50"
              >
                {isTraining ? "Computing Quantum Kernels..." : "Retrain QSVC Model"}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Confusion matrix breakdown ends */}
    </div>
  );
};
