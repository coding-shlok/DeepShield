"""Adaptive voting ensemble (PRD §4.2 / §6.1 / Contribution 3).

Members (ResNet+BiLSTM, EfficientNet, ViT, 3D-CNN) are frozen. Combination modes:
  * mean       – average of member probabilities
  * weighted   – learned global weights (softmax), fitted on the validation split
  * confidence – per-sample weights ∝ member confidence (1 - normalised entropy)
  * gated      – small MLP producing per-sample weights from member logits (adaptive voting)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEnsemble(nn.Module):
    def __init__(self, members: dict[str, nn.Module], mode: str = "gated", temperature: float = 1.0):
        super().__init__()
        self.members = nn.ModuleDict(members)
        for p in self.members.parameters():
            p.requires_grad_(False)
        n = len(members)
        self.mode = mode
        self.weights = nn.Parameter(torch.zeros(n))
        self.temperature = nn.Parameter(torch.full((n,), float(temperature)))  # per-member calibration
        self.gate = nn.Sequential(nn.Linear(2 * n, 32), nn.GELU(), nn.Linear(32, n))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.members.eval()  # frozen members always in eval mode (BN/dropout)
        return self

    def member_logits(self, x) -> torch.Tensor:
        outs = []
        for m in self.members.values():
            o = m(x)
            outs.append((o["logits"] if isinstance(o, dict) else o).float())
        return torch.stack(outs, 1)  # B,N,2

    def combine(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.temperature.clamp_min(0.05)[None, :, None]
        probs = logits.softmax(-1)
        if self.mode == "mean":
            w = torch.full(probs.shape[:2], 1.0 / probs.shape[1], device=probs.device)
        elif self.mode == "weighted":
            w = self.weights.softmax(0).expand(probs.shape[0], -1)
        elif self.mode == "confidence":
            ent = -(probs * probs.clamp_min(1e-8).log()).sum(-1) / torch.log(torch.tensor(2.0))
            w = (1 - ent).clamp_min(1e-3)
            w = w / w.sum(1, keepdim=True)
        elif self.mode == "gated":
            w = (self.weights[None] + self.gate(logits.flatten(1))).softmax(-1)
        else:
            raise ValueError(self.mode)
        p = (w[..., None] * probs).sum(1).clamp(1e-6, 1 - 1e-6)
        return p.log()  # log-probs act as logits for softmax / CE

    def forward(self, x):
        ml = self.member_logits(x)
        return {"logits": self.combine(ml), "member_logits": ml}


def fit_ensemble(ensemble: AdaptiveEnsemble, member_logits: torch.Tensor, labels: torch.Tensor,
                 epochs: int = 200, lr: float = 1e-2, weight_decay: float = 1e-3) -> AdaptiveEnsemble:
    """Fit combination parameters on cached validation-set member logits (cheap, members not re-run)."""
    if ensemble.mode == "mean":
        return ensemble
    params = [ensemble.weights, ensemble.temperature] + list(ensemble.gate.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.nll_loss(ensemble.combine(member_logits), labels)
        loss.backward()
        opt.step()
    return ensemble
