"""
app.py
==============================================================================
Breast Cancer Ultrasound Diagnostic System - Professional Web Application
==============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

import gradio as gr

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.hybrid_qnn import HybridQNN

# ==============================================================================
# Paths & constants
# ==============================================================================

CHECKPOINTS_DIR = _ROOT / "checkpoints"
CONFIG_PATH     = CHECKPOINTS_DIR / "model_config.json"
HYBRID_CKPT     = CHECKPOINTS_DIR / "best_hybrid_qnn.pt"

CLASS_NAMES  = ["Normal", "Benign", "Malignant"]
CLASS_COLORS = {"Normal": "#10b981", "Benign": "#f59e0b", "Malignant": "#ef4444"}
CLASS_BG     = {"Normal": "#064e3b", "Benign": "#78350f", "Malignant": "#7f1d1d"}
CLASS_ICON   = {"Normal": "🟢", "Benign": "🟡", "Malignant": "🔴"}
CLASS_DESC   = {
    "Normal":    "No structural abnormalities identified. Sonographic features consistent with normal tissue.",
    "Benign":    "Non-cancerous lesion or mass detected. Follow-up imaging and clinical observation recommended.",
    "Malignant": "Suspicious lesion detected with features suggestive of malignancy. Immediate specialist referral advised.",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ==============================================================================
# Model Cache
# ==============================================================================

_cache: dict = {}


def _load_model():
    if _cache:
        return _cache["model"], _cache["cfg"], _cache["device"]
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("model_config.json not found. Please train the model first.")
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    if not HYBRID_CKPT.exists():
        raise FileNotFoundError("Checkpoint not found. Please train the model first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HybridQNN(n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"])
    model.load_state_dict(torch.load(HYBRID_CKPT, map_location=device))
    model.to(device).eval()
    _cache.update({"model": model, "cfg": cfg, "device": device})
    return model, cfg, device


# ==============================================================================
# Inference Logic
# ==============================================================================

@torch.no_grad()
def predict(pil_img):
    """Main prediction entry point."""
    if pil_img is None:
        return (
            _placeholder_html(),
            _info_card_empty(),
        )

    try:
        model, cfg, device = _load_model()
    except FileNotFoundError as e:
        return (
            _error_html(str(e)),
            "<div style='color:#ef4444; padding:16px;'>System error: Model missing.</div>",
        )

    tensor = _TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(device)

    t0     = time.perf_counter()
    logits = model(tensor)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    probs    = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_cls = CLASS_NAMES[pred_idx]
    conf_pct = float(probs[pred_idx]) * 100.0

    result_html = _result_html(pred_cls, conf_pct, probs)
    info_html   = _info_card(lat_ms, pred_cls, conf_pct)

    return result_html, info_html


# ==============================================================================
# HTML Component Builders
# ==============================================================================

def _placeholder_html() -> str:
    return """
<div style='
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    min-height:360px; background:#1e293b;
    border:2px dashed #475569; border-radius:16px; padding:40px; text-align:center;
'>
    <div style='font-size:3.5em; margin-bottom:16px; opacity:0.8'>🩺</div>
    <div style='color:#ffffff; font-size:1.2em; font-weight:700; letter-spacing:0.02em'>
        Awaiting Ultrasound Image
    </div>
    <div style='color:#cbd5e1; font-size:0.92em; margin-top:8px; max-width:320px; line-height:1.5'>
        Upload a scan on the left and click <b style="color:#38bdf8">Analyze Scan</b> to generate diagnostic findings.
    </div>
</div>
"""


def _error_html(msg: str) -> str:
    return f"""
<div style='background:#450a0a; border:2px solid #ef4444; border-radius:16px; padding:24px;'>
    <div style='color:#ef4444; font-weight:700; font-size:1.1em; margin-bottom:8px'>
        ⚠️ System Error
    </div>
    <div style='color:#fca5a5; font-size:0.9em'>{msg}</div>
</div>
"""


def _result_html(pred_cls: str, conf_pct: float, probs: np.ndarray) -> str:
    color = CLASS_COLORS[pred_cls]
    bg    = CLASS_BG[pred_cls]
    icon  = CLASS_ICON[pred_cls]
    desc  = CLASS_DESC[pred_cls]

    bars = ""
    for cls, p in zip(CLASS_NAMES, probs):
        pct   = float(p) * 100.0
        c     = CLASS_COLORS[cls]
        bold  = "font-weight:700;" if cls == pred_cls else ""
        bars += f"""
        <div style='margin:12px 0'>
            <div style='display:flex; justify-content:space-between; font-size:0.92em; margin-bottom:6px'>
                <span style='color:#ffffff; {bold}'>{CLASS_ICON[cls]} {cls}</span>
                <span style='color:{c}; font-weight:700'>{pct:.1f}%</span>
            </div>
            <div style='background:#0f172a; border-radius:999px; height:12px; overflow:hidden'>
                <div style='
                    width:{pct:.1f}%;
                    background:linear-gradient(90deg, {c}aa, {c});
                    height:100%; border-radius:999px;
                '></div>
            </div>
        </div>
        """

    return f"""
<div style='
    background: linear-gradient(145deg, {bg}dd 0%, #1e293b 70%);
    border:2px solid {color}aa;
    border-radius:20px; padding:28px;
    font-family: Inter, sans-serif;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
'>
    <div style='text-align:center; margin-bottom:24px'>
        <div style='
            display:inline-block;
            background:{color}25; border:1px solid {color}88;
            border-radius:999px; padding:6px 18px;
            font-size:0.8em; font-weight:700;
            color:{color}; letter-spacing:0.08em;
            text-transform:uppercase; margin-bottom:14px;
        '>
            Diagnostic Outcome
        </div>
        <div style='font-size:3.5em; margin:4px 0'>{icon}</div>
        <div style='font-size:2.4em; font-weight:800; color:#ffffff; letter-spacing:-0.02em'>
            {pred_cls}
        </div>
        <div style='font-size:0.95em; color:#e2e8f0; margin-top:8px; line-height:1.5; max-width:340px; margin-left:auto; margin-right:auto'>
            {desc}
        </div>
    </div>

    <!-- Confidence Gauge -->
    <div style='
        background:#0f172a; border-radius:14px; padding:18px 20px;
        border:1px solid #334155; margin-bottom:20px;
    '>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px'>
            <span style='color:#cbd5e1; font-size:0.85em; font-weight:600; text-transform:uppercase; letter-spacing:0.05em'>
                Model Confidence
            </span>
            <span style='color:{color}; font-size:1.4em; font-weight:800'>{conf_pct:.1f}%</span>
        </div>
        <div style='background:#1e293b; border-radius:999px; height:12px; overflow:hidden'>
            <div style='
                width:{conf_pct:.1f}%;
                background:linear-gradient(90deg, {color}aa, {color});
                height:100%; border-radius:999px;
            '></div>
        </div>
    </div>

    <!-- Probability Breakdown -->
    <div style='
        background:#0f172a; border-radius:14px; padding:18px 20px;
        border:1px solid #334155;
    '>
        <div style='color:#cbd5e1; font-size:0.85em; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px'>
            Class Probability Distribution
        </div>
        {bars}
    </div>
</div>
"""


def _info_card_empty() -> str:
    return """
<div style='
    background:#1e293b; border:1px solid #334155;
    border-radius:16px; padding:28px 20px; text-align:center;
'>
    <div style='font-size:2.8em; margin-bottom:12px; opacity:0.8'>📊</div>
    <div style='color:#ffffff; font-weight:700; font-size:1.1em; margin-bottom:8px'>
        Clinical Summary
    </div>
    <div style='color:#cbd5e1; font-size:0.9em; line-height:1.6'>
        Upload an ultrasound scan and click <b style="color:#38bdf8">Analyze Scan</b> to display assessment metrics and recommendations.
    </div>
</div>
"""


def _info_card(lat_ms: float, pred_cls: str, conf_pct: float) -> str:
    risk_badge = {
        "Normal": "<span style='color:#10b981; font-weight:700'>🟢 LOW RISK</span>",
        "Benign": "<span style='color:#f59e0b; font-weight:700'>🟡 MODERATE RISK</span>",
        "Malignant": "<span style='color:#ef4444; font-weight:700'>🔴 HIGH RISK</span>",
    }[pred_cls]

    rec = {
        "Normal": "Routine screening according to standard clinical guidelines.",
        "Benign": "Follow-up ultrasound imaging in 6 months to monitor stability.",
        "Malignant": "Urgent referral to oncology / breast surgery specialist.",
    }[pred_cls]

    return f"""
<div style='
    background:#1e293b; border:1px solid #334155;
    border-radius:16px; padding:20px;
    font-family:Inter, sans-serif;
'>
    <div style='color:#38bdf8; font-weight:700; font-size:1.05em; margin-bottom:14px; text-transform:uppercase; letter-spacing:0.05em'>
        🏥 Clinical Assessment
    </div>

    <table style='width:100%; border-collapse:collapse; margin-bottom:16px;'>
        <tr style='border-bottom:1px solid #334155;'>
            <td style='padding:10px 0; color:#cbd5e1; font-size:0.9em;'>Primary Assessment</td>
            <td style='padding:10px 0; color:#ffffff; font-weight:700; text-align:right; font-size:0.95em;'>{pred_cls}</td>
        </tr>
        <tr style='border-bottom:1px solid #334155;'>
            <td style='padding:10px 0; color:#cbd5e1; font-size:0.9em;'>Risk Level</td>
            <td style='padding:10px 0; text-align:right; font-size:0.9em;'>{risk_badge}</td>
        </tr>
        <tr style='border-bottom:1px solid #334155;'>
            <td style='padding:10px 0; color:#cbd5e1; font-size:0.9em;'>Confidence Score</td>
            <td style='padding:10px 0; color:#ffffff; font-weight:700; text-align:right; font-size:0.95em;'>{conf_pct:.1f}%</td>
        </tr>
        <tr>
            <td style='padding:10px 0; color:#cbd5e1; font-size:0.9em;'>Scan Latency</td>
            <td style='padding:10px 0; color:#38bdf8; font-weight:600; text-align:right; font-size:0.9em;'>{lat_ms:.1f} ms</td>
        </tr>
    </table>

    <div style='background:#0f172a; border-left:4px solid #38bdf8; border-radius:8px; padding:12px 14px; margin-bottom:16px;'>
        <div style='color:#38bdf8; font-weight:700; font-size:0.82em; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;'>
            📋 Recommended Action
        </div>
        <div style='color:#ffffff; font-size:0.88em; line-height:1.5;'>
            {rec}
        </div>
    </div>

    <div style='color:#94a3b8; font-size:0.78em; line-height:1.5; font-style:italic;'>
        ℹ️ Clinical Note: Diagnostic assistance tool for research purposes. Findings require verification by a licensed radiologist.
    </div>
</div>
"""


# ==============================================================================
# CSS Customization - Absolute Dark Mode Theme Override
# ==============================================================================

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

body, .gradio-container, .gradio-container-5-0-0, [data-testid="app-element"] {
    background-color: #090d16 !important;
    background: #090d16 !important;
    color: #ffffff !important;
}

/* Force dark theme & remove stark white background on ALL Gradio blocks & widgets */
.block, .form, div[class*="block"], div[class*="form"], 
[data-testid="image-upload"], [data-testid="label"],
.output-class, .label-wrap, .panel, .header, label.block {
    background-color: #1e293b !important;
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #ffffff !important;
}

/* Completely strip out white header/label elements */
[data-testid="block-label"], .block-label, label span, span.subdued, .label-title {
    background-color: transparent !important;
    background: transparent !important;
    color: #cbd5e1 !important;
}

/* Upload Component styling */
.upload-box, [data-testid="image-upload"], .image-frame {
    background-color: #1e293b !important;
    border: 2px dashed #475569 !important;
    border-radius: 14px !important;
    color: #ffffff !important;
}

.upload-box:hover, [data-testid="image-upload"]:hover {
    border-color: #38bdf8 !important;
    background-color: #1e293b !important;
}

.upload-box span, .upload-box p, .upload-box div, .upload-box button {
    color: #ffffff !important;
}

/* Action Button */
.action-btn {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    border-radius: 12px !important;
    padding: 14px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
    transition: all 0.2s ease !important;
}

.action-btn:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
}

/* Tab Navigation Styling */
.tab-nav {
    border-bottom: 2px solid #334155 !important;
}

.tab-nav button {
    font-weight: 600 !important;
    font-size: 1em !important;
    color: #cbd5e1 !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
}

.tab-nav button.selected {
    color: #38bdf8 !important;
    background-color: #1e293b !important;
    border-bottom: 3px solid #38bdf8 !important;
    font-weight: 700 !important;
}

.tab-nav button:hover {
    color: #ffffff !important;
}

footer { display: none !important; }
"""

# ==============================================================================
# Header Component
# ==============================================================================

_HEADER = """
<div style='
    text-align: center;
    padding: 32px 20px 24px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 18px;
    border: 1px solid #334155;
    margin-bottom: 20px;
'>
    <div style='
        display:inline-block;
        background:#1e3a8a; border: 1px solid #3b82f6;
        border-radius: 999px; padding: 5px 18px;
        font-size: 0.8em; color: #93c5fd;
        font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; margin-bottom: 12px;
    '>
        Medical AI Diagnostic Assistant
    </div>

    <h1 style='
        margin: 0 0 8px;
        font-size: 2.3em;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
    '>
        Breast Cancer Ultrasound Classifier
    </h1>

    <p style='color: #cbd5e1; margin: 0; font-size: 0.98em;'>
        Automated 3-Class Ultrasound Image Analysis & Clinical Risk Stratification
    </p>
</div>
"""

# ==============================================================================
# Gradio Interface
# ==============================================================================

def build_ui() -> gr.Blocks:
    checkpoints_ok = CONFIG_PATH.exists() and HYBRID_CKPT.exists()

    with gr.Blocks(title="Breast Cancer Ultrasound Classifier") as demo:

        gr.HTML(_HEADER)

        if not checkpoints_ok:
            gr.Markdown(
                "## ⚠️  Model Not Trained Yet\n\n"
                "Please run `python src/train.py` to train the classifier before launching the application."
            )
            return demo

        with gr.Tabs():

            # TAB 1 — Diagnostic Interface
            with gr.Tab("🩺 Clinical Diagnosis"):
                with gr.Row(equal_height=False):

                    # Left Column — Input
                    with gr.Column(scale=4, min_width=300):
                        gr.HTML("""
                        <div style='color:#38bdf8; font-size:0.9em; font-weight:700;
                                    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px'>
                            📷 ULTRASOUND IMAGE UPLOAD
                        </div>
                        """)
                        img_input = gr.Image(
                            type="pil",
                            label="",
                            show_label=False,
                            height=290,
                            image_mode="RGB",
                            elem_classes=["upload-box"],
                        )
                        run_btn = gr.Button(
                            "🔍  Analyze Scan",
                            size="lg",
                            elem_classes=["action-btn"],
                        )
                        gr.HTML("""
                        <div style='
                            background:#1e293b; border:1px solid #334155;
                            border-radius:12px; padding:16px; margin-top:14px;
                            font-size:0.88em; color:#f1f5f9; line-height:1.7;
                        '>
                            <div style='color:#38bdf8; font-weight:700; margin-bottom:6px; font-size:0.95em'>
                                📋 Workflow Instructions
                            </div>
                            <div style='margin-bottom:4px'><b style='color:#38bdf8'>1.</b> Upload breast ultrasound scan (PNG/JPG)</div>
                            <div style='margin-bottom:4px'><b style='color:#38bdf8'>2.</b> Click <b style='color:#60a5fa'>Analyze Scan</b> button</div>
                            <div><b style='color:#38bdf8'>3.</b> Review findings & clinical recommendations</div>
                        </div>
                        """)

                    # Center Column — Primary Output Card
                    with gr.Column(scale=5, min_width=360):
                        gr.HTML("""
                        <div style='color:#38bdf8; font-size:0.9em; font-weight:700;
                                    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px'>
                            🔬 DIAGNOSTIC FINDINGS
                        </div>
                        """)
                        result_html = gr.HTML(_placeholder_html())

                    # Right Column — Summary & Recommendations
                    with gr.Column(scale=3, min_width=240):
                        gr.HTML("""
                        <div style='color:#38bdf8; font-size:0.9em; font-weight:700;
                                    text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px'>
                            📋 CLINICAL SUMMARY
                        </div>
                        """)
                        info_html = gr.HTML(_info_card_empty())

            # TAB 2 — System Overview
            with gr.Tab("ℹ️ System Overview"):
                gr.HTML("""
<div style='background:#1e293b; border:1px solid #334155; border-radius:16px; padding:24px; color:#f1f5f9;'>
    <h3 style='color:#38bdf8; margin-top:0; font-size:1.2em;'>🧠 System Architecture</h3>
    <p style='color:#cbd5e1; line-height:1.6;'>
        The Breast Cancer Ultrasound Classifier combines a ResNet-18 feature extraction backbone with a 4-qubit Variational Quantum Circuit trained to perform robust 3-class lesion identification on breast ultrasound images.
    </p>

    <hr style='border:none; border-top:1px solid #334155; margin:20px 0;'>

    <h3 style='color:#38bdf8; font-size:1.2em;'>🏷️ Diagnostic Classification Framework</h3>

    <table style='width:100%; border-collapse:collapse; margin-top:12px;'>
        <thead>
            <tr style='background:#0f172a; color:#38bdf8;'>
                <th style='padding:12px; border:1px solid #334155; text-align:left;'>Category</th>
                <th style='padding:12px; border:1px solid #334155; text-align:left;'>Sonographic Features</th>
                <th style='padding:12px; border:1px solid #334155; text-align:left;'>Standard Management</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style='padding:12px; border:1px solid #334155; color:#10b981; font-weight:700;'>🟢 Normal</td>
                <td style='padding:12px; border:1px solid #334155; color:#e2e8f0;'>Uniform echotexture, no focal mass or structural distortion.</td>
                <td style='padding:12px; border:1px solid #334155; color:#e2e8f0;'>Routine clinical screening</td>
            </tr>
            <tr>
                <td style='padding:12px; border:1px solid #334155; color:#f59e0b; font-weight:700;'>🟡 Benign</td>
                <td style='padding:12px; border:1px solid #334155; color:#e2e8f0;'>Circumscribed margin, oval shape, parallel orientation (e.g. fibroadenoma/cyst).</td>
                <td style='padding:12px; border:1px solid #334155; color:#e2e8f0;'>6-month surveillance imaging</td>
            </tr>
            <tr>
                <td style='padding:12px; border:1px solid #334155; color:#ef4444; font-weight:700;'>🔴 Malignant</td>
                <td style='padding:12px; border:1px solid #334155; color:#e2e8f0;'>Spiculated/microlobulated margin, non-parallel, acoustic shadowing.</td>
                <td style='padding:12px; border:1px solid #334155; color:#e2e8f0;'>Urgent biopsy & oncology consult</td>
            </tr>
        </tbody>
    </table>

    <hr style='border:none; border-top:1px solid #334155; margin:20px 0;'>

    <div style='color:#94a3b8; font-size:0.85em; font-style:italic;'>
        ℹ️ Regulatory Disclaimer: Intended solely for academic, clinical evaluation, and research demonstration.
    </div>
</div>
""")

        # Event wiring
        outputs = [result_html, info_html]
        run_btn.click(fn=predict, inputs=img_input, outputs=outputs)
        img_input.change(fn=predict, inputs=img_input, outputs=outputs)

    return demo


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Breast Cancer Ultrasound Diagnostic Web Application")
    print("=" * 60)

    try:
        model, cfg, device = _load_model()
        print(f"[app] Model loaded successfully on device: {device}")
    except FileNotFoundError as e:
        print(f"[app] WARNING: {e}")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
        theme=gr.themes.Base(),
        css=_CSS,
    )
