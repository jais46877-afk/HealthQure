"""
src/models/hybrid_qnn.py
==============================================================================
Hybrid Quantum-Classical Neural Network  — v2 (Accuracy Boost Edition)
==============================================================================

Architecture v2 changes:
  1. Data Re-Uploading: features re-injected before EACH quantum layer
     instead of once at the start → vastly richer Hilbert space mapping
  2. 6 Qubits default (2⁶=64 Hilbert dimensions, up from 2⁴=16)
  3. Dual Pauli Measurement: ⟨Z_i⟩ AND ⟨X_i⟩ → 2×n_qubits output features
  4. Selective layer4 unfreezing in ResNet-18 for ultrasound domain adaptation
  5. Larger input support (256×256 from data_loader v2)
==============================================================================
"""

import math

import torch
import torch.nn as nn
import torchvision.models as tv_models

import pennylane as qml


# ==============================================================================
# Backbone — ResNet-18 with selective layer4 fine-tuning
# ==============================================================================

def _resnet18_backbone(unfreeze_layer4: bool = True) -> nn.Module:
    """
    Load pre-trained ResNet-18, remove fc head.
    - Layers 1–3 remain frozen (universal edge/texture detectors)
    - Layer 4 optionally unfrozen with LR=1e-5 for domain adaptation
    """
    backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False

    # Selectively unfreeze the final conv block (layer4) for ultrasound adaptation
    if unfreeze_layer4:
        for param in backbone.layer4.parameters():
            param.requires_grad = True
        print("[HybridQNN] ResNet-18 layer4 UNFROZEN for domain fine-tuning.")
    else:
        print("[HybridQNN] ResNet-18 fully frozen (transfer learning only).")

    backbone.fc = nn.Identity()    # output: (B, 512)
    return backbone


# ==============================================================================
# Quantum Circuit with Data Re-Uploading
# ==============================================================================

def _build_v1_qnode(n_qubits: int, n_layers: int):
    """v1 Standard Variational Quantum Circuit (Single Z-measurement)."""
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff_method = "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        diff_method = "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return dev, circuit


def _build_reupload_qnode(n_qubits: int, n_layers: int):
    """Data Re-Uploading Variational Quantum Circuit (v2)."""
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff_method = "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        diff_method = "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for layer_idx in range(n_layers):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(
                weights[layer_idx:layer_idx + 1], wires=range(n_qubits)
            )
        z_meas = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        x_meas = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        return z_meas + x_meas

    return dev, circuit


# ==============================================================================
# Main Model Class
# ==============================================================================

class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network supporting v1 and v2.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 3,
        n_classes: int = 3,
        unfreeze_layer4: bool = True,
        version: str = "v2",
    ) -> None:
        super().__init__()

        self.n_qubits        = n_qubits
        self.n_layers        = n_layers
        self.n_classes       = n_classes
        self.unfreeze_layer4 = unfreeze_layer4
        self.version         = version

        # 1. ResNet-18 backbone
        self.backbone = _resnet18_backbone(unfreeze_layer4=unfreeze_layer4)
        self._pi_scale = math.pi

        if version == "v1":
            # Classical bottleneck v1: 512 → 64 → n_qubits
            self.bottleneck = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, n_qubits),
                nn.Tanh(),
            )
            _dev, _circuit = _build_v1_qnode(n_qubits, n_layers)
            self._qdev     = _dev
            weight_shapes  = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer    = qml.qnn.TorchLayer(_circuit, weight_shapes=weight_shapes)
            self.classifier = nn.Linear(n_qubits, n_classes)
        else:
            # Classical bottleneck v2: 512 → 128 → n_qubits
            self._q_out_dim = 2 * n_qubits
            self.bottleneck = nn.Sequential(
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_qubits),
                nn.Tanh(),
            )
            _dev, _circuit = _build_reupload_qnode(n_qubits, n_layers)
            self._qdev     = _dev
            weight_shapes  = {"weights": (n_layers, n_qubits, 3)}
            self.qlayer    = qml.qnn.TorchLayer(_circuit, weight_shapes=weight_shapes)
            self.classifier = nn.Sequential(
                nn.Linear(self._q_out_dim, 16),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(16, n_classes),
            )

    # --------------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (B, 3, H, W)   — H/W = 256 recommended

        Returns
        -------
        logits : Tensor shape (B, n_classes)
        """
        # Classical feature extraction
        features = self.backbone(x)                              # (B, 512)
        angles   = self.bottleneck(features) * self._pi_scale   # (B, n_qubits)

        # Data re-uploading quantum layer
        q_out    = self.qlayer(angles)                           # (B, 2*n_qubits)

        # Classification head
        logits   = self.classifier(q_out)                        # (B, 3)
        return logits

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """Return trainable and total parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def circuit_depth(self) -> int:
        return self.n_layers

    def __repr__(self) -> str:
        info = self.count_parameters()
        return (
            "HybridQNN v2("
            "n_qubits={}, n_layers={}, n_classes={}, "
            "re_uploading=True, dual_pauli=True, "
            "trainable_params={:,})".format(
                self.n_qubits, self.n_layers, self.n_classes, info["trainable"]
            )
        )


# ==============================================================================
# Smoke-test
# ==============================================================================

if __name__ == "__main__":
    model = HybridQNN(n_qubits=6, n_layers=3)
    print(model)
    p = model.count_parameters()
    print("Trainable : {:,}  |  Total : {:,}".format(p["trainable"], p["total"]))

    dummy = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print("Output shape  : {}".format(out.shape))
    print("Output logits :", out)
