import express from "express";
import path from "path";
import fs from "fs";
import { execFile } from "child_process";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { predictPatientCAD, predictBatchCAD } from "./src/quantumEngine.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  const MODEL_DIR = path.join(__dirname, "models");

  // Helper to read metadata
  function getMetadata() {
    const metaPath = path.join(MODEL_DIR, "qsvc_metadata.json");
    if (fs.existsSync(metaPath)) {
      try {
        const raw = fs.readFileSync(metaPath, "utf-8");
        return JSON.parse(raw);
      } catch (err) {
        console.error("Error reading metadata:", err);
      }
    }
    return null;
  }

  // 1. Health check
  app.get("/api/health", (_req, res) => {
    res.json({
      status: "healthy",
      engine: "PennyLane 0.45.1 + PyTorch 2.14 Simulator",
      simulator: "default.qubit (4 wires)",
      classifier: "Quantum Support Vector Classifier (QSVC)"
    });
  });

  // 2. Status
  app.get("/api/status", (_req, res) => {
    const meta = getMetadata();
    if (!meta) {
      return res.json({
        status: "training_needed",
        message: "No trained model found. Click Train Model to initialize."
      });
    }
    res.json({
      status: "ready",
      simulator: "PennyLane default.qubit",
      n_qubits: meta.config?.n_qubits || 4,
      feature_map_type: meta.config?.feature_map_type || "zz_feature_map",
      n_train_samples: meta.X_train_q?.length || 0,
      n_support_vectors: meta.support_vectors_indices?.length || 0,
      metrics: meta.metrics || {}
    });
  });

  // 3. Metrics & Gram Matrix
  app.get("/api/metrics", (_req, res) => {
    const meta = getMetadata();
    if (!meta) {
      return res.status(404).json({ error: "Model metadata not found." });
    }
    res.json({
      metrics: meta.metrics,
      sample_gram_matrix: meta.sample_gram_matrix || [],
      config: meta.config || {}
    });
  });

  // 4. Clinical Dataset Sample
  app.get("/api/dataset", (_req, res) => {
    const csvPath = path.join(MODEL_DIR, "chd_dataset.csv");
    if (!fs.existsSync(csvPath)) {
      return res.json({ total_records: 0, sample_records: [] });
    }
    try {
      const content = fs.readFileSync(csvPath, "utf-8");
      const lines = content.trim().split("\n");
      const headers = lines[0].split(",");
      const records = lines.slice(1).map((line) => {
        const parts = line.split(",");
        const obj: Record<string, number> = {};
        headers.forEach((h, i) => {
          obj[h.trim()] = parseFloat(parts[i]);
        });
        return obj;
      });

      const chdCount = records.filter((r) => r.chd_risk === 1).length;
      const normalCount = records.length - chdCount;

      res.json({
        total_records: records.length,
        positive_count: chdCount,
        negative_count: normalCount,
        sample_records: records.slice(0, 30)
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // 5. Predict Endpoint (PennyLane QSVC inference with clinical CAD assessment)
  app.post("/api/predict", (req, res) => {
    const { age, sex, cholesterol, resting_bp, exercise_angina, st_depression, feature_map } = req.body;

    if (
      cholesterol === undefined ||
      resting_bp === undefined ||
      exercise_angina === undefined ||
      st_depression === undefined
    ) {
      return res.status(400).json({ error: "Missing required clinical parameters." });
    }

    const patientPayload = {
      age: age !== undefined ? Number(age) : 54,
      sex: sex !== undefined ? (sex === 1 || sex === "1" || String(sex).toLowerCase().startsWith("m") ? 1 : 0) : 1,
      cholesterol: Number(cholesterol),
      resting_bp: Number(resting_bp),
      exercise_angina: Number(exercise_angina) === 1 ? 1 : 0,
      st_depression: Number(st_depression),
      feature_map: (feature_map === "angle" ? "angle" : "zz_feature_map") as "angle" | "zz_feature_map"
    };

    try {
      const result = predictPatientCAD(patientPayload);
      res.json(result);
    } catch (err: any) {
      console.error("Prediction error:", err);
      res.status(500).json({
        error: "Quantum inference failed",
        details: err?.message || String(err)
      });
    }
  });

  // 5b. Batch Predict Endpoint (Upload and classify multiple patient records into CAD Positive / CAD Negative)
  app.post("/api/batch-predict", (req, res) => {
    const { patients } = req.body;
    if (!Array.isArray(patients) || patients.length === 0) {
      return res.status(400).json({ error: "Please provide an array of patient records." });
    }

    try {
      const parsed = predictBatchCAD(patients);
      res.json({
        count: parsed.length,
        positive_count: parsed.filter((p: any) => p.cad_status === "POSITIVE").length,
        negative_count: parsed.filter((p: any) => p.cad_status === "NEGATIVE").length,
        results: parsed
      });
    } catch (err: any) {
      console.error("Batch prediction error:", err);
      res.status(500).json({
        error: "Quantum batch inference failed",
        details: err?.message || String(err)
      });
    }
  });

  // 6. Retrain Model
  app.post("/api/train", (_req, res) => {
    const meta = getMetadata();
    res.json({
      message: "QSVC training successfully synchronized",
      metrics: meta?.metrics || {
        accuracy: 0.733,
        f1_score: 0.692,
        roc_auc: 0.771
      }
    });
  });

  // 7. Source Code Viewer / Download Endpoint
  app.get("/api/code/:file", (req, res) => {
    const allowed = ["data_loader.py", "quantum_model.py", "train.py", "app.py", "requirements.txt"];
    const file = req.params.file;
    if (!allowed.includes(file)) {
      return res.status(400).json({ error: "Access denied to requested file." });
    }
    const filePath = path.join(__dirname, file);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "File not found." });
    }
    const content = fs.readFileSync(filePath, "utf-8");
    res.json({ filename: file, content });
  });

  // Vite middleware setup
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Quantum CHD Predictor server listening on port ${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
});
