"""DeepShieldNet: ResNet-50 + BiLSTM + attention fusion with multi-task heads (PRD §6.1).

Input : x in [0,1], shape [B,T,3,H,W] (video clip) or [B,3,H,W] (single selfie, T=1).
        Normalisation to [-1,1] happens inside the model (adversarial eps stays in pixel units).
Output: dict(
    logits         [B,2]  real/fake (softmax -> deepfake probability),
    frame_logits   [B,T,2] per-frame predictions (temporal-consistency loss, frame-level scores),
    method_logits  [B,M]  manipulation method (auxiliary task),
    temporal_score [B]    temporal anomaly logit from the BiLSTM stream,
    embedding      [B,D]  fused embedding (contrastive loss / ensembles))
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.constants import METHODS
from .backbones import SpatialBackbone
from .fusion import AttentionFusion
from .temporal import build_temporal


class InputNormalize(nn.Module):
    """[0,1] -> [-1,1] (PRD Input Module)."""

    def forward(self, x):
        return x * 2.0 - 1.0


class DeepShieldNet(nn.Module):
    def __init__(self, backbone: dict | None = None, temporal: str | None = "bilstm", temporal_kw: dict | None = None,
                 fusion: dict | None = None, num_methods: int = len(METHODS), dropout: float = 0.3,
                 frame_chunk: int | None = None):
        super().__init__()
        self.normalize = InputNormalize()
        self.backbone = SpatialBackbone(**(backbone or {}))
        D = self.backbone.out_dim
        self.temporal = build_temporal(temporal, D, **(temporal_kw or {}))
        Dt = self.temporal.out_dim if self.temporal is not None else None
        self.fusion = AttentionFusion(D, Dt, **(fusion or {}))
        F = self.fusion.out_dim
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(F, F // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(F // 2, 2))
        self.frame_head = nn.Linear(D, 2)
        self.method_head = nn.Linear(F, num_methods)
        self.temporal_head = nn.Linear(Dt, 1) if Dt else None
        self.frame_chunk = frame_chunk  # limit memory: run backbone on chunks of frames

    # -- stages are separate methods so the latency profiler can time them (PRD §6.4)
    def extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        flat = self.normalize(x.flatten(0, 1))
        if self.frame_chunk and flat.shape[0] > self.frame_chunk:
            feats = torch.cat([self.backbone(c) for c in flat.split(self.frame_chunk)], 0)
        else:
            feats = self.backbone(flat)  # all frames in one vectorised forward pass
        return feats.view(B, T, -1)

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        f = self.extract_spatial(x)
        h = self.temporal(f) if self.temporal is not None else None
        z, _ = self.fusion(f, h)
        z = self.dropout(z)
        out = {
            "logits": self.head(z),
            "frame_logits": self.frame_head(f),
            "method_logits": self.method_head(z),
            "embedding": z,
        }
        if h is not None:
            out["temporal_score"] = self.temporal_head(h.mean(1)).squeeze(-1)
        return out


class LogitsOnly(nn.Module):
    """Wraps a dict-output model so attacks / ONNX export see a plain logits tensor."""

    def __init__(self, model: nn.Module, key: str = "logits"):
        super().__init__()
        self.model, self.key = model, key

    def forward(self, x):
        out = self.model(x)
        return out[self.key] if isinstance(out, dict) else out


class ProbOutput(nn.Module):
    """Deployment wrapper: returns P(fake) [B] and logits [B,2]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = LogitsOnly(self.model)(x)
        return torch.softmax(logits.float(), -1)[:, 1], logits
