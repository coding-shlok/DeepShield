"""Multi-head self-attention fusion of spatial + temporal tokens (PRD §4.1 Ensemble Classifier)."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, spatial_dim: int, temporal_dim: int | None, dim: int = 512, heads: int = 4,
                 layers: int = 1, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.sp = nn.Linear(spatial_dim, dim)
        self.tp = nn.Linear(temporal_dim, dim) if temporal_dim else None
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, dim))
        self.type_emb = nn.Parameter(torch.zeros(2, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor | None = None):
        """spatial [B,T,Ds], temporal [B,T,Dt] -> (fused [B,dim], attention-pooled tokens)."""
        B, T, _ = spatial.shape
        pos = self.pos[:, 1:T + 1]
        tokens = [self.sp(spatial) + pos + self.type_emb[0]]
        if temporal is not None and self.tp is not None:
            tokens.append(self.tp(temporal) + pos + self.type_emb[1])
        x = torch.cat([self.cls.expand(B, -1, -1) + self.pos[:, :1]] + tokens, 1)
        x = self.encoder(x)
        return self.norm(x[:, 0]), x[:, 1:]
