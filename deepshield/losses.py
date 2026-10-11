"""Loss functions (PRD §6.2-6.3).

Primary : focal loss  FL(p_t) = -α_t (1-p_t)^γ log(p_t), γ=2 (class imbalance 80:20)
Auxiliary:
  * manipulation-method cross entropy (multi-task)
  * temporal consistency  L_temporal = MSE(p_t, p_{t+1}) on per-frame fake probabilities
  * contrastive (real/fake separation in embedding space). NOTE: the PRD writes
    ||f(x_real) - f(x_fake)||²; minimising that literally would *merge* the classes, so we
    implement the standard margin form (Hadsell et al. 2006): pull same-class pairs together
    and push real-fake pairs at least `margin` apart.
  * adversarial loss: same objective evaluated on FGSM/PGD examples (computed in the trainer)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.constants import IGNORE_INDEX


class FocalLoss(nn.Module):
    """Softmax focal loss over 2 classes supporting soft (mixup) targets.

    alpha: per-class weights [α_real, α_fake]. With an 80:20 real:fake prior, α ≈ [0.25, 0.75]
    up-weights the minority (fake) class.
    """

    def __init__(self, gamma: float = 2.0, alpha=(0.25, 0.75), label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None)
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        C = logits.shape[-1]
        if target.dim() == logits.dim() - 1:
            target = F.one_hot(target.long(), C).float()
        if self.ls:
            target = target * (1 - self.ls) + self.ls / C
        logp = F.log_softmax(logits, -1)
        p = logp.exp()
        loss = -target * (1 - p).pow(self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)
        return loss.sum(-1).mean()


def temporal_consistency_loss(frame_logits: torch.Tensor) -> torch.Tensor:
    """MSE between consecutive per-frame fake probabilities, frame_logits [B,T,2]."""
    if frame_logits is None or frame_logits.shape[1] < 2:
        return frame_logits.new_zeros(()) if frame_logits is not None else torch.zeros(())
    p = frame_logits.float().softmax(-1)[..., 1]
    return F.mse_loss(p[:, 1:], p[:, :-1])


def contrastive_margin_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    if labels.dim() > 1:  # soft labels from mixup -> skip pairs that are ambiguous
        labels = labels.argmax(-1)
    z = F.normalize(emb.float(), dim=-1)
    d = torch.cdist(z, z)  # in [0, 2]
    same = (labels[:, None] == labels[None, :]).float()
    eye = torch.eye(len(z), device=z.device)
    pos = same - eye
    neg = 1 - same
    l_pos = (pos * d.pow(2)).sum() / pos.sum().clamp_min(1)
    l_neg = (neg * F.relu(margin - d).pow(2)).sum() / neg.sum().clamp_min(1)
    return l_pos + l_neg


class MultiTaskLoss(nn.Module):
    """Weighted sum of the PRD's multi-task objectives. Returns (total, parts dict)."""

    def __init__(self, focal: dict | None = None, w_method: float = 0.2, w_temporal: float = 0.1,
                 w_contrastive: float = 0.05, w_frame: float = 0.3, w_temporal_score: float = 0.1,
                 contrastive_margin: float = 1.0):
        super().__init__()
        self.focal = FocalLoss(**(focal or {}))
        self.w = dict(method=w_method, temporal=w_temporal, contrastive=w_contrastive, frame=w_frame,
                      temporal_score=w_temporal_score)
        self.margin = contrastive_margin

    def forward(self, out: dict, target: torch.Tensor, method: torch.Tensor | None = None,
                method_b: torch.Tensor | None = None, lam: float = 1.0) -> tuple[torch.Tensor, dict]:
        parts = {"focal": self.focal(out["logits"], target)}
        if self.w["frame"] and out.get("frame_logits") is not None:
            fl = out["frame_logits"]
            T = fl.shape[1]
            tgt = target if target.dim() == 2 else F.one_hot(target, 2).float()
            parts["frame"] = self.focal(fl.flatten(0, 1), tgt.repeat_interleave(T, 0))
        if self.w["temporal"] and out.get("frame_logits") is not None:
            parts["temporal"] = temporal_consistency_loss(out["frame_logits"])
        if self.w["method"] and method is not None and out.get("method_logits") is not None:
            ml = out["method_logits"].float()
            lm = F.cross_entropy(ml, method, ignore_index=IGNORE_INDEX)
            if method_b is not None and lam < 1.0:
                lm = lam * lm + (1 - lam) * F.cross_entropy(ml, method_b, ignore_index=IGNORE_INDEX)
            parts["method"] = torch.nan_to_num(lm)
        if self.w["temporal_score"] and out.get("temporal_score") is not None:
            soft = target[:, 1] if target.dim() == 2 else target.float()
            parts["temporal_score"] = F.binary_cross_entropy_with_logits(out["temporal_score"].float(), soft)
        if self.w["contrastive"] and out.get("embedding") is not None and len(target) > 1:
            parts["contrastive"] = contrastive_margin_loss(out["embedding"], target, self.margin)
        total = parts["focal"] + sum(self.w[k] * v for k, v in parts.items() if k != "focal")
        return total, {k: float(v.detach()) for k, v in parts.items()}


def distillation_loss(student_logits, teacher_logits, target, T: float = 4.0, alpha: float = 0.7,
                      hard_loss: nn.Module | None = None) -> torch.Tensor:
    """Hinton KD: α·T²·KL(softmax(t/T) || softmax(s/T)) + (1-α)·hard loss."""
    kd = F.kl_div(F.log_softmax(student_logits.float() / T, -1), F.softmax(teacher_logits.float() / T, -1),
                  reduction="batchmean") * T * T
    hard = (hard_loss or FocalLoss())(student_logits, target)
    return alpha * kd + (1 - alpha) * hard
