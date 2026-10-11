"""White/black-box L∞ attacks in pixel space [0,1] (PRD §3.1 O3, §8.2: eps = 8/255).

Models are any callables returning logits [B,2] or a dict with 'logits'. Inputs can be
images [B,3,H,W] or clips [B,T,3,H,W]; the perturbation has the same shape as the input.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logits(model, x):
    out = model(x)
    return (out["logits"] if isinstance(out, dict) else out).float()


@torch.enable_grad()
def fgsm(model, x, y, eps: float = 8 / 255) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(_logits(model, x), y)
    g, = torch.autograd.grad(loss, x)
    return (x + eps * g.sign()).clamp(0, 1).detach()


@torch.enable_grad()
def pgd(model, x, y, eps: float = 8 / 255, alpha: float | None = None, steps: int = 10,
        random_start: bool = True) -> torch.Tensor:
    alpha = alpha or max(eps / 4, 2.5 * eps / steps)
    x0 = x.detach()
    adv = x0 + torch.empty_like(x0).uniform_(-eps, eps) if random_start else x0.clone()
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(_logits(model, adv), y)
        g, = torch.autograd.grad(loss, adv)
        adv = adv.detach() + alpha * g.sign()
        adv = torch.min(torch.max(adv, x0 - eps), x0 + eps).clamp(0, 1)
    return adv.detach()


@torch.enable_grad()
def apgd_ce(model, x, y, eps: float = 8 / 255, steps: int = 50, rho: float = 0.75) -> torch.Tensor:
    """Auto-PGD with CE loss (Croce & Hein, 2020): step-size free, momentum, checkpoints."""
    x0 = x.detach()
    B = x0.shape[0]
    shape = (B,) + (1,) * (x0.dim() - 1)
    checkpoints, p = [0], [0.0, 0.22]
    while p[-1] < 1:
        p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    checkpoints = sorted(set(int(math.ceil(q * steps)) for q in p if q <= 1))

    def loss_grad(z):
        z = z.clone().requires_grad_(True)
        l = F.cross_entropy(_logits(model, z), y, reduction="none")
        g, = torch.autograd.grad(l.sum(), z)
        return l.detach(), g.detach()

    step = torch.full(shape, 2 * eps, device=x0.device)
    adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    loss, grad = loss_grad(adv)
    best_loss, best_adv = loss.clone(), adv.clone()
    prev = adv.clone()
    succ_count = torch.zeros(B, device=x0.device)
    last_ck_loss = best_loss.clone()
    last_ck_step = step.clone()
    for k in range(steps):
        z = adv + step * grad.sign()
        z = torch.min(torch.max(z, x0 - eps), x0 + eps).clamp(0, 1)
        a = 0.75 if k > 0 else 1.0
        new = adv + a * (z - adv) + (1 - a) * (adv - prev)
        new = torch.min(torch.max(new, x0 - eps), x0 + eps).clamp(0, 1)
        prev, adv = adv, new
        loss, grad = loss_grad(adv)
        improved = loss > best_loss
        succ_count += improved.float()
        best_loss = torch.where(improved, loss, best_loss)
        best_adv[improved] = adv[improved]
        if k + 1 in checkpoints and k > 0:
            h = checkpoints[checkpoints.index(k + 1) - 1] if checkpoints.index(k + 1) > 0 else 0
            window = max(k + 1 - h, 1)
            cond1 = succ_count < rho * window
            cond2 = (last_ck_step.view(-1) == step.view(-1)) & (last_ck_loss == best_loss)
            reduce = cond1 | cond2
            last_ck_step, last_ck_loss = step.clone(), best_loss.clone()
            step[reduce.view(shape[:1])] /= 2
            adv[reduce] = best_adv[reduce]
            if reduce.any():
                _, grad_r = loss_grad(adv)
                grad[reduce] = grad_r[reduce]
            succ_count.zero_()
    return best_adv.detach()


@torch.no_grad()
def square_attack(model, x, y, eps: float = 8 / 255, queries: int = 500, p_init: float = 0.05) -> torch.Tensor:
    """Score-based black-box Square Attack (Andriushchenko et al., 2020), L∞, margin loss.
    Gradient-free: guards against gradient masking giving a false sense of robustness."""
    x0 = x.detach()
    img_shape = x0.shape[-3:]
    c, h, w = img_shape
    lead = x0.shape[:-3]

    def margin(z):
        lg = _logits(model, z)
        true = lg.gather(1, y[:, None]).squeeze(1)
        other = lg.clone()
        other.scatter_(1, y[:, None], float("-inf"))
        return true - other.max(1).values  # <0 => misclassified

    # vertical stripes init
    adv = (x0 + eps * torch.randint(0, 2, (*lead, c, 1, w), device=x0.device).float().mul(2).sub(1)).clamp(0, 1)
    m = margin(adv)
    for i in range(queries):
        active = m > 0
        if not active.any():
            break
        it = int(i / max(queries, 1) * 10000)  # original schedule is defined for 10k queries
        halvings = sum(it > b for b in (10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000))
        frac = p_init / (2 ** halvings)
        s = max(int(round(math.sqrt(frac * h * w))), 1)
        r, cc = torch.randint(0, h - s + 1, (1,)).item(), torch.randint(0, w - s + 1, (1,)).item()
        cand = adv.clone()
        delta = eps * torch.randint(0, 2, (*lead, c, 1, 1), device=x0.device).float().mul(2).sub(1)
        cand[..., r:r + s, cc:cc + s] = (x0[..., r:r + s, cc:cc + s] + delta).clamp(0, 1)
        cand = torch.min(torch.max(cand, x0 - eps), x0 + eps)
        mc = margin(cand)
        better = (mc < m) & active
        adv[better] = cand[better]
        m = torch.where(better, mc, m)
    return adv


def run_autoattack(model, x, y, eps: float = 8 / 255, version: str = "custom", square_queries: int = 1000):
    """AutoAttack. For a 2-class problem the targeted DLR/FAB components of the 'standard' suite are
    undefined (they need >= 3 classes), so we run APGD-CE + Square – the untargeted part of the suite.
    Uses the official `autoattack` package when installed, otherwise the built-in implementations."""
    wrapped = _LogitModule(model)
    try:
        from autoattack import AutoAttack

        aa = AutoAttack(wrapped, norm="Linf", eps=eps, version="custom", attacks_to_run=["apgd-ce", "square"],
                        verbose=False)
        aa.square.n_queries = square_queries
        if x.dim() == 5:  # official AA assumes images: flatten frames into channels
            B, T = x.shape[:2]
            flat = _FlatClip(wrapped, T)
            aa.model = flat
            adv = aa.run_standard_evaluation(x.flatten(1, 2), y, bs=x.shape[0])
            return adv.view_as(x)
        return aa.run_standard_evaluation(x, y, bs=x.shape[0])
    except ImportError:
        adv = apgd_ce(wrapped, x, y, eps)
        with torch.no_grad():
            still = _logits(wrapped, adv).argmax(1) == y
        if still.any():
            adv[still] = square_attack(wrapped, x[still], y[still], eps, queries=square_queries)
        return adv


class _LogitModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return _logits(self.model, x)


class _FlatClip(nn.Module):
    def __init__(self, model, T):
        super().__init__()
        self.model, self.T = model, T

    def forward(self, x):
        B = x.shape[0]
        return self.model(x.view(B, self.T, 3, *x.shape[-2:]))


ATTACKS = {"fgsm": fgsm, "pgd": pgd, "apgd": apgd_ce, "square": square_attack, "autoattack": run_autoattack}
