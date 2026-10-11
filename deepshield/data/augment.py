"""KYC-specific capture simulation (PRD §5.3 / §8.3) and standard augmentations.

All transforms operate on *clips*: a list of HxWx3 uint8 RGB numpy frames. Random
parameters are sampled once per clip so a perturbation is temporally consistent
(a real phone camera does not change JPEG quality every frame) – except hand-tremor
jitter which is deliberately per-frame.

Output of the full transform: float tensor [T, 3, S, S] in [0, 1]. The model performs
normalisation to [-1, 1] internally so that adversarial budgets are expressed in pixel
space (eps = 8/255).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np
import torch

Clip = list  # list[np.ndarray] (H, W, 3) uint8 RGB


# ----------------------------------------------------------------------------- primitives
def degrade_resolution(img: np.ndarray, scale: float) -> np.ndarray:
    """Downscale then upscale back: 1080p -> 360p corresponds to scale = 1/3."""
    h, w = img.shape[:2]
    sh, sw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    if length <= 1:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k)


def jitter(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def cast_shadow(img: np.ndarray, strength: float, angle_deg: float, offset: float) -> np.ndarray:
    """Linear-gradient shadow (e.g. side lighting at a window)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle_deg)
    proj = (xx / w - 0.5) * math.cos(a) + (yy / h - 0.5) * math.sin(a) + offset
    mask = 1.0 - strength * np.clip(proj * 2.0, 0, 1)
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def occlude(img: np.ndarray, kind: str, rng: random.Random) -> np.ndarray:
    """Partial occlusions typical for selfies: medical mask, glasses, beard shadow, hand/random patch."""
    out = img.copy()
    h, w = img.shape[:2]
    if kind == "mask":
        top = int(h * rng.uniform(0.55, 0.65))
        color = rng.choice([(230, 230, 235), (90, 140, 200), (40, 40, 40)])
        pts = np.array([[int(w * 0.15), top], [int(w * 0.85), top], [int(w * 0.75), int(h * 0.95)],
                        [int(w * 0.25), int(h * 0.95)]], np.int32)
        cv2.fillPoly(out, [pts], color)
    elif kind == "glasses":
        y = int(h * rng.uniform(0.36, 0.44))
        r = int(w * 0.11)
        for cx in (int(w * 0.33), int(w * 0.67)):
            cv2.circle(out, (cx, y), r, (20, 20, 20), max(2, w // 64))
            overlay = out.copy()
            cv2.circle(overlay, (cx, y), r, (10, 10, 10), -1)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        cv2.line(out, (int(w * 0.33) + r, y), (int(w * 0.67) - r, y), (20, 20, 20), max(2, w // 64))
    elif kind == "beard_shadow":
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (w // 2, int(h * 0.85)), (int(w * 0.32), int(h * 0.2)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), w / 20)
        out = np.clip(out.astype(np.float32) * (1 - 0.5 * mask[..., None]), 0, 255).astype(np.uint8)
    else:  # random patch (hand, phone edge)
        ph, pw = int(h * rng.uniform(0.15, 0.35)), int(w * rng.uniform(0.15, 0.35))
        y0, x0 = rng.randint(0, h - ph), rng.randint(0, w - pw)
        out[y0:y0 + ph, x0:x0 + pw] = rng.randint(0, 255)
    return out


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- KYC perturbation
@dataclass
class KYCPerturbation:
    """Configurable mobile-capture simulator.

    Each field is either None (disabled), a fixed value, or a (min, max) range sampled per clip.
    `p_*` probabilities apply only when `random=True` (training). With `random=False` every
    enabled perturbation is applied deterministically (evaluation presets).
    """

    resolution_scale: float | tuple | None = (0.25, 0.6)   # 360p/1080p = 0.33
    blur_px: int | tuple | None = (3, 5)                    # motion blur kernel length
    jitter_px: float | None = 5.0                           # +/- hand tremor translation
    jpeg_quality: int | tuple | None = (60, 80)
    brightness: float | tuple | None = (0.7, 1.3)           # +/-30%
    shadow: float | tuple | None = (0.2, 0.5)
    occlusion: Sequence[str] | str | None = ("mask", "glasses", "beard_shadow", "patch")
    noise_sigma: float | tuple | None = (2.0, 8.0)
    random: bool = True
    p: dict = field(default_factory=lambda: dict(resolution=0.4, blur=0.3, jitter=0.3, jpeg=0.6,
                                                  brightness=0.5, shadow=0.2, occlusion=0.15, noise=0.2))
    seed: int | None = None

    def _val(self, v, rng):
        if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            lo, hi = v
            return rng.randint(lo, hi) if isinstance(lo, int) and isinstance(hi, int) else rng.uniform(lo, hi)
        return v

    def _on(self, key, value, rng) -> bool:
        if value is None:
            return False
        return (not self.random) or rng.random() < self.p.get(key, 0.0)

    def __call__(self, clip: Clip) -> Clip:
        rng = random.Random(self.seed) if self.seed is not None else random
        nrng = np.random.default_rng(self.seed)
        ops: list[Callable[[np.ndarray, int], np.ndarray]] = []

        if self._on("resolution", self.resolution_scale, rng):
            s = self._val(self.resolution_scale, rng)
            ops.append(lambda im, i, s=s: degrade_resolution(im, s))
        if self._on("jitter", self.jitter_px, rng):
            j = float(self.jitter_px)
            shifts = [(rng.uniform(-j, j), rng.uniform(-j, j)) for _ in clip]
            ops.append(lambda im, i, sh=shifts: jitter(im, *sh[i]))
        if self._on("blur", self.blur_px, rng):
            L, ang = int(self._val(self.blur_px, rng)), rng.uniform(0, 180)
            ops.append(lambda im, i, L=L, a=ang: motion_blur(im, L, a))
        if self._on("brightness", self.brightness, rng):
            b = self._val(self.brightness, rng)
            ops.append(lambda im, i, b=b: adjust_brightness(im, b))
        if self._on("shadow", self.shadow, rng):
            st, ang, off = self._val(self.shadow, rng), rng.uniform(0, 360), rng.uniform(-0.3, 0.3)
            ops.append(lambda im, i, st=st, a=ang, o=off: cast_shadow(im, st, a, o))
        if self._on("occlusion", self.occlusion, rng):
            kinds = [self.occlusion] if isinstance(self.occlusion, str) else list(self.occlusion)
            kind = rng.choice(kinds)
            occ_seed = rng.randint(0, 2**31 - 1)
            ops.append(lambda im, i, k=kind, s=occ_seed: occlude(im, k, random.Random(s)))
        if self._on("noise", self.noise_sigma, rng):
            sg = self._val(self.noise_sigma, rng)
            ops.append(lambda im, i, sg=sg: gaussian_noise(im, sg, nrng))
        # JPEG last: transmission over WhatsApp/Telegram happens after capture.
        if self._on("jpeg", self.jpeg_quality, rng):
            q = int(self._val(self.jpeg_quality, rng))
            ops.append(lambda im, i, q=q: jpeg_compress(im, q))

        out = []
        for i, im in enumerate(clip):
            for op in ops:
                im = op(im, i)
            out.append(im)
        return out


def _only(**kw) -> KYCPerturbation:
    base = dict(resolution_scale=None, blur_px=None, jitter_px=None, jpeg_quality=None, brightness=None,
                shadow=None, occlusion=None, noise_sigma=None, random=False, seed=0)
    base.update(kw)
    return KYCPerturbation(**base)


# Deterministic evaluation presets (PRD §8.3). Seeds fixed for reproducibility.
KYC_EVAL_PRESETS: dict[str, KYCPerturbation | None] = {
    "clean": None,
    "res_360p": _only(resolution_scale=1 / 3),
    "motion_blur_5px": _only(blur_px=5, jitter_px=5.0),
    "jpeg_q70": _only(jpeg_quality=70),
    "jpeg_q60": _only(jpeg_quality=60),
    "brightness_minus30": _only(brightness=0.7),
    "brightness_plus30": _only(brightness=1.3),
    "shadow": _only(shadow=0.4),
    "occlusion_mask": _only(occlusion="mask"),
    "occlusion_glasses": _only(occlusion="glasses"),
    "sensor_noise": _only(noise_sigma=6.0),
    "combined": _only(resolution_scale=1 / 3, blur_px=5, jitter_px=5.0, jpeg_quality=70, brightness=0.7),
}


# ----------------------------------------------------------------------------- full pipelines
class ClipTransform:
    """Geometric + photometric + optional KYC perturbation, then to tensor [T,3,S,S] in [0,1]."""

    def __init__(self, size: int = 224, train: bool = True, kyc: KYCPerturbation | None = None,
                 hflip: float = 0.5, scale: tuple = (0.85, 1.0), color_jitter: float = 0.2,
                 grayscale_p: float = 0.05):
        self.size = size
        self.train = train
        self.kyc = kyc
        self.hflip = hflip
        self.scale = scale
        self.color_jitter = color_jitter
        self.grayscale_p = grayscale_p

    def _geometric(self, clip: Clip) -> Clip:
        h, w = clip[0].shape[:2]
        if self.train:
            s = random.uniform(*self.scale)
            ch, cw = int(h * s), int(w * s)
            y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
            flip = random.random() < self.hflip
        else:
            ch, cw, y0, x0, flip = h, w, 0, 0, False
        out = []
        for im in clip:
            im = im[y0:y0 + ch, x0:x0 + cw]
            if flip:
                im = im[:, ::-1]
            im = cv2.resize(np.ascontiguousarray(im), (self.size, self.size), interpolation=cv2.INTER_AREA
                            if ch > self.size else cv2.INTER_LINEAR)
            out.append(im)
        return out

    def _photometric(self, clip: Clip) -> Clip:
        if not self.train or self.color_jitter <= 0:
            return clip
        cj = self.color_jitter
        b, c, s = (random.uniform(1 - cj, 1 + cj) for _ in range(3))
        gray = random.random() < self.grayscale_p
        out = []
        for im in clip:
            f = im.astype(np.float32) * b
            mean = f.mean()
            f = (f - mean) * c + mean
            g = f.mean(axis=2, keepdims=True)
            f = g + (f - g) * (0.0 if gray else s)
            out.append(np.clip(f, 0, 255).astype(np.uint8))
        return out

    def __call__(self, clip: Clip) -> torch.Tensor:
        if self.kyc is not None:
            clip = self.kyc(clip)
        clip = self._geometric(clip)
        clip = self._photometric(clip)
        arr = np.stack(clip).astype(np.float32) / 255.0  # T,H,W,3
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def build_train_transform(cfg: dict | None = None) -> ClipTransform:
    cfg = dict(cfg or {})
    size = cfg.get("image_size", 224)
    kyc_cfg = cfg.get("kyc", {"enabled": True})
    kyc = None
    if kyc_cfg and kyc_cfg.get("enabled", True):
        kw = {k: (tuple(v) if isinstance(v, list) and k != "occlusion" else v)
              for k, v in kyc_cfg.items() if k != "enabled"}
        kyc = KYCPerturbation(**kw)
    return ClipTransform(size=size, train=True, kyc=kyc, hflip=cfg.get("hflip", 0.5),
                         scale=tuple(cfg.get("scale", (0.85, 1.0))), color_jitter=cfg.get("color_jitter", 0.2))


def build_eval_transform(size: int = 224, preset: str | None = None) -> ClipTransform:
    kyc = KYC_EVAL_PRESETS[preset] if preset else None
    return ClipTransform(size=size, train=False, kyc=kyc)
