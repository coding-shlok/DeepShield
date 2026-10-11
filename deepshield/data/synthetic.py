"""Procedural synthetic 'face' videos for smoke tests and CI.

Real clips: smooth, temporally coherent faces. Fake clips carry artefacts that real
detectors look for: blending boundary around the inner face, GAN-like periodic
up-sampling pattern, colour mismatch and temporal flicker. Not a substitute for real
data – it only verifies the full pipeline (prepare -> train -> eval -> export) runs.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .manifest import assign_splits, write_manifest

FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _face(rng: np.random.Generator, size: int, t: int, params: dict) -> np.ndarray:
    img = np.full((size, size, 3), params["bg"], np.uint8)
    cx = size // 2 + int(3 * np.sin(t / 4 + params["phase"]))
    cy = size // 2 + int(2 * np.cos(t / 5 + params["phase"]))
    cv2.ellipse(img, (cx, cy), (int(size * 0.3), int(size * 0.4)), 0, 0, 360, params["skin"], -1)
    blink = (t + params["blink"]) % 12 == 0
    for dx in (-1, 1):
        ex, ey = cx + dx * int(size * 0.12), cy - int(size * 0.08)
        if blink:
            cv2.line(img, (ex - 8, ey), (ex + 8, ey), (40, 30, 30), 2)
        else:
            cv2.circle(img, (ex, ey), max(3, size // 32), (40, 30, 30), -1)
    mouth_open = int(4 + 3 * np.sin(t / 3 + params["phase"]))
    cv2.ellipse(img, (cx, cy + int(size * 0.17)), (int(size * 0.09), mouth_open), 0, 0, 360, (120, 40, 50), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _manipulate(img: np.ndarray, rng: np.random.Generator, t: int, method: str, p: dict) -> np.ndarray:
    size = img.shape[0]
    out = img.astype(np.float32)
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(mask, (size // 2, size // 2), (int(size * 0.22), int(size * 0.28)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 2)
    inner = out.copy()
    # colour mismatch of swapped region (+ temporal flicker)
    inner += np.array(p["tint"]) * (1.0 + 0.6 * np.sin(t * 1.7 + p["phase"]))
    if method in ("Deepfakes", "FaceSwap"):  # periodic GAN-style upsampling grid
        yy, xx = np.mgrid[0:size, 0:size]
        inner += p["grid_amp"] * (((yy // 2 + xx // 2) % 2) * 2 - 1)[..., None]
    if method in ("Face2Face", "NeuralTextures"):  # over-smooth texture
        inner = cv2.GaussianBlur(inner, (0, 0), 1.5)
    out = out * (1 - mask[..., None]) + inner * mask[..., None]
    # boundary artefact
    edge = cv2.Canny((mask * 255).astype(np.uint8), 50, 150) > 0
    out[edge] += p["edge"]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_synthetic_dataset(root: str | Path, n_videos: int = 120, n_frames: int = 16, size: int = 128,
                           fake_ratio: float = 0.5, seed: int = 0, name: str = "synthetic") -> Path:
    root = Path(root)
    (root / "faces").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    n_ids = max(4, n_videos // 4)
    for v in range(n_videos):
        label = int(rng.random() < fake_ratio)
        identity = int(rng.integers(n_ids))
        idrng = np.random.default_rng(identity + 1000 * seed)
        params = dict(bg=tuple(int(x) for x in idrng.integers(20, 200, 3)),
                      skin=tuple(int(x) for x in idrng.integers(120, 230, 3)),
                      phase=float(rng.uniform(0, 6.28)), blink=int(rng.integers(12)))
        method = rng.choice(FAKE_METHODS) if label else ""
        fp = dict(tint=rng.uniform(-14, 14, 3).tolist(), grid_amp=float(rng.uniform(5, 10)),
                  edge=float(rng.uniform(15, 35)), phase=float(rng.uniform(0, 6.28)))
        sid = f"{name}_{v:05d}"
        d = root / "faces" / sid
        d.mkdir(exist_ok=True)
        for t in range(n_frames):
            im = _face(rng, size, t, params)
            if label:
                im = _manipulate(im, rng, t, method, fp)
            cv2.imwrite(str(d / f"{t:03d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(sample_id=sid, dataset=name, label=label, method=method, group=f"id{identity}",
                         path=f"faces/{sid}", n_frames=n_frames, split=""))
    df = assign_splits(pd.DataFrame(rows), seed=seed)
    write_manifest(df, root / "manifest.csv")
    return root / "manifest.csv"
