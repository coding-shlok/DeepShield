"""Per-stage latency profiling (PRD §6.4 table) and model size."""
from __future__ import annotations

import io
import time

import numpy as np
import torch


def _sync(device):
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()


def _time(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        _sync(device)
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(np.mean(ts))}


def model_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


@torch.no_grad()
def profile_model(model, device="cuda", clip_len: int = 16, image_size: int = 224, batch: int = 1,
                  iters: int = 50, warmup: int = 10, half: bool | None = None) -> dict:
    """Times the full forward and, for DeepShieldNet, each stage separately."""
    from ..models.detector import DeepShieldNet

    model = model.eval().to(device)
    half = (device == "cuda") if half is None else half
    x = torch.rand(batch, clip_len, 3, image_size, image_size, device=device)
    ctx = torch.autocast("cuda", dtype=torch.float16) if half and device == "cuda" else torch.autocast("cpu", enabled=False)
    res = {"device": str(device), "batch": batch, "clip_len": clip_len, "size_mb_fp32": model_size_mb(model)}
    with ctx:
        res["total"] = _time(lambda: model(x), device, iters, warmup)
        if isinstance(model, DeepShieldNet):
            from ..data.augment import build_eval_transform  # noqa: F401  (preprocessing timed below)

            f = model.extract_spatial(x)
            h = model.temporal(f) if model.temporal is not None else None
            z, _ = model.fusion(f, h)
            res["spatial"] = _time(lambda: model.extract_spatial(x), device, iters, warmup)
            if model.temporal is not None:
                res["temporal"] = _time(lambda: model.temporal(f), device, iters, warmup)
            res["fusion"] = _time(lambda: model.fusion(f, h), device, iters, warmup)
            res["classify"] = _time(lambda: model.head(z).softmax(-1), device, iters, warmup)
    # preprocessing: resize + normalise 16 face crops (detection excluded, PRD budget 5 ms)
    import cv2

    frames = [np.random.randint(0, 255, (256, 256, 3), np.uint8) for _ in range(clip_len)]

    def prep():
        arr = np.stack([cv2.resize(f, (image_size, image_size)) for f in frames]).astype(np.float32) / 255
        return torch.from_numpy(arr).permute(0, 3, 1, 2)

    res["preprocess_cpu"] = _time(prep, "cpu", max(iters // 2, 5), 2)
    res["ms"] = res["total"]["median_ms"]
    return res
