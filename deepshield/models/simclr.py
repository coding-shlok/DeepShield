"""SimCLR contrastive pretraining for identity-aware representations (PRD §6.2 Phase 1)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import SpatialBackbone
from .detector import InputNormalize


class SimCLR(nn.Module):
    def __init__(self, backbone: dict, proj_dim: int = 128, hidden: int = 2048):
        super().__init__()
        self.normalize = InputNormalize()
        self.backbone = SpatialBackbone(**backbone)
        d = self.backbone.out_dim
        self.projector = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                       nn.Linear(hidden, proj_dim))

    def forward(self, x):
        return self.projector(self.backbone(self.normalize(x)))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Normalised temperature-scaled cross entropy over 2N views (in-batch negatives)."""
    z = F.normalize(torch.cat([z1, z2], 0).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)
