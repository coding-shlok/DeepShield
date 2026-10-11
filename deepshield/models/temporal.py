"""Temporal consistency modules (PRD §4.2 'Temporal Consistency Analysis')."""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMTemporal(nn.Module):
    """2-layer BiLSTM (hidden 512) over per-frame embeddings -> per-frame temporal features."""

    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = 2 * hidden

    def forward(self, f: torch.Tensor) -> torch.Tensor:  # [B,T,D] -> [B,T,2H]
        self.lstm.flatten_parameters()
        out, _ = self.lstm(self.in_norm(f))
        return out


class TemporalConvTemporal(nn.Module):
    """Lightweight 1D temporal conv alternative (faster, ONNX/TensorRT friendly)."""

    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 2, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        mods, c = [], in_dim
        for _ in range(layers):
            mods += [nn.Conv1d(c, hidden, kernel, padding=kernel // 2), nn.BatchNorm1d(hidden), nn.GELU(),
                     nn.Dropout(dropout)]
            c = hidden
        self.net = nn.Sequential(*mods)
        self.out_dim = hidden

    def forward(self, f):
        return self.net(f.transpose(1, 2)).transpose(1, 2)


class FrameDiffEncoder(nn.Module):
    """Encodes first-order temporal differences of embeddings (frame jitter / flicker cue)."""

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU())
        self.out_dim = out_dim

    def forward(self, f):
        d = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)
        return self.proj(d)


def build_temporal(kind: str | None, in_dim: int, **kw) -> nn.Module | None:
    if kind in (None, "none"):
        return None
    if kind == "bilstm":
        return BiLSTMTemporal(in_dim, **kw)
    if kind == "tconv":
        return TemporalConvTemporal(in_dim, **kw)
    raise ValueError(kind)
