"""Video/image -> aligned face crops on disk.

For each video we sample `n_frames` frames uniformly (PRD: 16 keyframes at inference; we
store 32 by default so training can pick random temporal windows), detect the largest
face, smooth the box over time, expand by `margin`, square it and save JPEG crops.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .face_detect import FaceDetector


def read_video_frames(path: str | Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:  # some containers don't report length – decode everything
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return []
        idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).round().astype(int)
        return [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx]
    wanted = set(np.linspace(0, total - 1, min(n_frames, total)).round().astype(int).tolist())
    i = 0
    last = max(wanted)
    while i <= last:
        ok = cap.grab()
        if not ok:
            break
        if i in wanted:
            ok, f = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def _largest(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]) * d[1])[0]


def square_crop(img: np.ndarray, box: np.ndarray, margin: float, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * margin
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad = max(0, -x1, -y1, x2 - w, y2 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR)


def crop_faces(frames: Iterable[np.ndarray], detector: FaceDetector, margin: float = 1.3, size: int = 256,
               smooth: float = 0.6, min_score: float = 0.5) -> list[np.ndarray]:
    crops, prev = [], None
    if getattr(detector, "name", "") == "none":
        margin = 1.0  # frames are already face crops: no expansion (would add black borders)
    for f in frames:
        dets = [d for d in detector.detect(f) if d[1] >= min_score]
        box = _largest(dets)
        if box is None:
            if prev is None:
                continue  # no face yet
            box = prev
        elif prev is not None:
            box = smooth * prev + (1 - smooth) * box  # temporal smoothing reduces crop jitter
        prev = box
        crops.append(square_crop(f, box, margin, size))
    return crops


def save_crops(crops: list[np.ndarray], out_dir: Path, quality: int = 95) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(crops):
        cv2.imwrite(str(out_dir / f"{i:03d}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
    return len(crops)


def process_video(path: str | Path, out_dir: Path, detector: FaceDetector, n_frames: int = 32,
                  margin: float = 1.3, size: int = 256) -> int:
    frames = read_video_frames(path, n_frames)
    return save_crops(crop_faces(frames, detector, margin, size), out_dir)


def process_images(paths: list, out_dir: Path, detector: FaceDetector, margin: float = 1.3,
                   size: int = 256, n_frames: int | None = None) -> int:
    """Image sequence (or single image) -> crops. `paths` may hold file paths or encoded bytes."""
    if n_frames and len(paths) > n_frames:
        idx = np.linspace(0, len(paths) - 1, n_frames).round().astype(int)
        paths = [paths[i] for i in idx]
    frames = []
    for p in paths:
        if isinstance(p, (bytes, bytearray)):
            im = cv2.imdecode(np.frombuffer(p, np.uint8), cv2.IMREAD_COLOR)
        else:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return save_crops(crop_faces(frames, detector, margin, size), out_dir)
