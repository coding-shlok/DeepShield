"""Robustness filter (PRD §4.1): perturbation / zero-day attack detection and a
robustness-confidence score attached to every prediction.

* Feature squeezing (Xu et al., NDSS 2018): adversarial perturbations are brittle under
  bit-depth reduction, median smoothing and JPEG re-encoding; a large L1 change in the output
  distribution flags the input. Threshold calibrated on clean validation data at a target FPR.
* robustness_confidence: agreement rate of predictions under small Gaussian noise (a cheap
  local-stability estimate in [0,1]; 1 = stable decision).
"""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _probs(model, x):
    out = model(x)
    return (out["logits"] if isinstance(out, dict) else out).float().softmax(-1)


def bit_depth(x, bits: int = 5):
    m = 2 ** bits - 1
    return torch.round(x * m) / m


def median_filter(x, k: int = 3):
    shp = x.shape
    flat = x.reshape(-1, *shp[-3:])
    pad = k // 2
    p = F.pad(flat, (pad, pad, pad, pad), mode="reflect")
    patches = p.unfold(2, k, 1).unfold(3, k, 1)
    return patches.contiguous().flatten(-2).median(-1).values.reshape(shp)


def jpeg_squeeze(x, quality: int = 75):
    shp = x.shape
    flat = x.reshape(-1, *shp[-3:]).detach().cpu()
    outs = []
    for im in flat:
        arr = (im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        rec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).astype(np.float32) / 255
        outs.append(torch.from_numpy(rec).permute(2, 0, 1))
    return torch.stack(outs).reshape(shp).to(x.device)


class FeatureSqueezingDetector:
    def __init__(self, model, squeezers=("bit5", "median3", "jpeg75"), threshold: float | None = None):
        self.model = model
        self.squeezers = squeezers
        self.threshold = threshold

    def _squeeze(self, name, x):
        if name.startswith("bit"):
            return bit_depth(x, int(name[3:]))
        if name.startswith("median"):
            return median_filter(x, int(name[6:]))
        if name.startswith("jpeg"):
            return jpeg_squeeze(x, int(name[4:]))
        raise ValueError(name)

    @torch.no_grad()
    def score(self, x) -> torch.Tensor:
        p = _probs(self.model, x)
        return torch.stack([(p - _probs(self.model, self._squeeze(s, x))).abs().sum(-1)
                            for s in self.squeezers], 0).max(0).values

    @torch.no_grad()
    def calibrate(self, clean_scores: np.ndarray, fpr: float = 0.05) -> float:
        self.threshold = float(np.quantile(clean_scores, 1 - fpr))
        return self.threshold

    def detect(self, x) -> torch.Tensor:
        assert self.threshold is not None, "call calibrate() first"
        return self.score(x) > self.threshold


@torch.no_grad()
def robustness_confidence(model, x, sigma: float = 0.03, n: int = 8, threshold: float = 0.5) -> torch.Tensor:
    base = _probs(model, x)[:, 1] >= threshold
    agree = torch.zeros(base.shape, dtype=torch.float32, device=base.device)
    for _ in range(n):
        noisy = _probs(model, (x + sigma * torch.randn_like(x)).clamp(0, 1))[:, 1] >= threshold
        agree += (noisy == base).float()
    return agree / n
