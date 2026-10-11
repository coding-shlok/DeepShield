"""Randomized smoothing certification (Cohen, Rosenfeld & Kolter, ICML 2019) – PRD §4.2 / D4.

g(x) = argmax_c P(f(x + δ) = c), δ ~ N(0, σ²I). If the lower confidence bound p_A > 1/2 then
g is certifiably constant within L2 radius R = σ Φ⁻¹(p_A).

Note: an L∞ budget ε corresponds to an L2 ball of radius ε·sqrt(d); for 224² RGB images that is
far beyond what smoothing can certify, so we report certified accuracy at L2 radii and empirical
L∞ robustness separately. Train with `train.gaussian_noise_sigma` for good smoothed accuracy.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import beta, norm


def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


class Smooth:
    ABSTAIN = -1

    def __init__(self, model, num_classes: int = 2, sigma: float = 0.25, batch: int = 64):
        self.model, self.C, self.sigma, self.batch = model, num_classes, sigma, batch

    @torch.no_grad()
    def _counts(self, x: torch.Tensor, n: int) -> np.ndarray:
        counts = np.zeros(self.C, dtype=np.int64)
        remaining = n
        while remaining > 0:
            b = min(self.batch, remaining)
            xb = x.unsqueeze(0).expand(b, *x.shape)
            noisy = xb + torch.randn_like(xb) * self.sigma
            out = self.model(noisy)
            pred = (out["logits"] if isinstance(out, dict) else out).argmax(1)
            counts += np.bincount(pred.cpu().numpy(), minlength=self.C)
            remaining -= b
        return counts

    def predict(self, x: torch.Tensor, n: int = 100, alpha: float = 0.001) -> int:
        from scipy.stats import binomtest

        c = self._counts(x, n)
        top2 = c.argsort()[::-1][:2]
        na, nb = c[top2[0]], c[top2[1]]
        if binomtest(int(na), int(na + nb), 0.5).pvalue > alpha:
            return self.ABSTAIN
        return int(top2[0])

    def certify(self, x: torch.Tensor, n0: int = 100, n: int = 1000, alpha: float = 0.001) -> tuple[int, float]:
        """Returns (prediction or ABSTAIN, certified L2 radius) for a single input x (no batch dim)."""
        c0 = self._counts(x, n0)
        ca = int(c0.argmax())
        c = self._counts(x, n)
        pa = _clopper_pearson_lower(int(c[ca]), n, alpha)
        if pa < 0.5:
            return self.ABSTAIN, 0.0
        return ca, float(self.sigma * norm.ppf(pa))


def certified_accuracy(preds: np.ndarray, radii: np.ndarray, labels: np.ndarray,
                       thresholds=(0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)) -> dict:
    correct = preds == labels
    return {f"cert_acc@{r}": float(np.mean(correct & (radii >= r))) for r in thresholds}
