"""3D-CNN spatiotemporal branch (PRD §6.1 ensemble member) – torchvision R(2+1)D / R3D."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.constants import METHODS


class Video3DNet(nn.Module):
    def __init__(self, arch_name: str = "r2plus1d_18", pretrained: bool = True, num_methods: int = len(METHODS),
                 dropout: float = 0.3):
        super().__init__()
        from torchvision.models import video as tvv

        weights = {"r3d_18": "R3D_18_Weights", "r2plus1d_18": "R2Plus1D_18_Weights", "mc3_18": "MC3_18_Weights"}[arch_name]
        w = getattr(tvv, weights).DEFAULT if pretrained else None
        try:
            self.net = getattr(tvv, arch_name)(weights=w)
        except Exception as e:  # offline cluster nodes
            print(f"[video3d] pretrained weights unavailable ({e}); random init")
            self.net = getattr(tvv, arch_name)(weights=None)
        dim = self.net.fc.in_features
        self.net.fc = nn.Identity()
        # Kinetics normalisation
        self.register_buffer("mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 2)
        self.method_head = nn.Linear(dim, num_methods)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if x.shape[1] < 4:  # 3D conv needs a few frames; repeat stills
            x = x.repeat(1, 4 // x.shape[1] + 1, 1, 1, 1)[:, :4]
        v = (x.permute(0, 2, 1, 3, 4) - self.mean) / self.std  # B,3,T,H,W
        z = self.net(v)
        zd = self.dropout(z)
        return {"logits": self.head(zd), "method_logits": self.method_head(zd), "embedding": z}
