"""Production inference pipeline (PRD §6.4, D5).

    pred = DeepShieldPredictor("runs/stage3/best.pth")           # PyTorch backend
    pred = DeepShieldPredictor("outputs/export/deepshield.onnx")  # ONNX Runtime backend
    pred.predict("selfie.jpg") / pred.predict("liveness.mp4") / pred.predict_batch([...])

Pipeline: decode -> sample 16 keyframes -> face detection & crop -> resize 224 -> [0,1] tensor
-> model -> P(fake) -> KYC decision with a manual-review band.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from ..data.extract import crop_faces, read_video_frames
from ..data.face_detect import build_detector

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class Prediction:
    path: str
    prob_fake: float
    label: str               # "real" | "fake"
    decision: str            # "accept" | "reject" | "manual_review" | "no_face"
    confidence: float        # distance from the decision threshold, in [0,1]
    n_frames: int
    frame_probs: list = field(default_factory=list)
    robustness_confidence: float | None = None
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class DeepShieldPredictor:
    def __init__(self, model_path: str, device: str | None = None, clip_len: int | None = None,
                 image_size: int | None = None,
                 detector: str = "retinaface", threshold: float | None = None, review_band: float = 0.15,
                 robustness_check: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_len, self.size = clip_len or 16, image_size or 224
        self.threshold, self.review_band = threshold, review_band  # band = +/- around threshold
        self.robustness_check = robustness_check
        self.detector = build_detector(detector, self.device)
        self.backend = "onnx" if str(model_path).endswith(".onnx") else "torch"
        if self.backend == "onnx":
            import onnxruntime as ort

            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(str(model_path), providers=prov)
            shp = self.sess.get_inputs()[0].shape
            if isinstance(shp[1], int):
                self.clip_len = shp[1]  # static clip length / size baked into the graph
            if isinstance(shp[-1], int):
                self.size = shp[-1]
        else:
            from ..models.factory import load_model

            self.model = load_model(model_path).to(self.device).eval()
            if self.threshold is None:
                self.threshold = self.model.calibrated_threshold
            dc = getattr(self.model, "data_config", {})  # match the training input format
            self.clip_len = clip_len or dc.get("clip_len", 16)
            self.size = image_size or dc.get("image_size", 224)
        if self.threshold is None:  # ONNX: read calibrated threshold from export metadata if present
            meta = Path(model_path).with_suffix(".json")
            self.threshold = float(__import__("json").loads(meta.read_text()).get("threshold", 0.5)) \
                if meta.exists() else 0.5

    # ------------------------------------------------------------------ preprocessing
    def _load_frames(self, path: str) -> list[np.ndarray]:
        if Path(path).suffix.lower() in VIDEO_EXT:
            return read_video_frames(path, self.clip_len)
        im = cv2.imread(str(path))
        if im is None:
            raise FileNotFoundError(path)
        return [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)]

    def preprocess(self, frames: list[np.ndarray], detect: bool = True) -> torch.Tensor | None:
        crops = crop_faces(frames, self.detector) if detect else frames
        if not crops:
            return None
        arr = np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_AREA) for c in crops])
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2)  # T,3,H,W
        if self.backend == "onnx" and x.shape[0] != self.clip_len:  # static graph: repeat / subsample
            idx = np.linspace(0, x.shape[0] - 1, self.clip_len).round().astype(int)
            x = x[idx]
        return x

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
        if self.backend == "onnx":
            p, _ = self.sess.run(["prob_fake", "logits"], {"frames": x.numpy()})
            return p, None
        out = self.model(x.to(self.device))
        p = out["logits"].float().softmax(-1)[:, 1].cpu().numpy()
        fp = out.get("frame_logits")
        return p, (fp.float().softmax(-1)[..., 1].cpu().numpy() if fp is not None else None)

    def _decide(self, p: float) -> str:
        if abs(p - self.threshold) <= self.review_band:
            return "manual_review"
        return "reject" if p >= self.threshold else "accept"

    def _conf(self, p: float) -> float:
        """Distance from the decision threshold, scaled to [0,1]."""
        t = self.threshold
        return float((p - t) / (1 - t) if p >= t else (t - p) / t) if 0 < t < 1 else abs(p - 0.5) * 2

    def predict_frames(self, frames: list[np.ndarray], path: str = "<frames>", detect: bool = True) -> Prediction:
        t0 = time.perf_counter()
        x = self.preprocess(frames, detect)
        t1 = time.perf_counter()
        if x is None:
            return Prediction(path, float("nan"), "unknown", "no_face", 0.0, 0)
        p, fp = self._forward(x[None])
        t2 = time.perf_counter()
        prob = float(p[0])
        rc = None
        if self.robustness_check and self.backend == "torch":
            from ..robustness.detection import robustness_confidence

            rc = float(robustness_confidence(self.model, x[None].to(self.device), threshold=self.threshold)[0])
        return Prediction(path, prob, "fake" if prob >= self.threshold else "real", self._decide(prob),
                          self._conf(prob), int(x.shape[0]), fp[0].tolist() if fp is not None else [], rc,
                          {"preprocess": (t1 - t0) * 1000, "model": (t2 - t1) * 1000})

    def predict(self, path: str) -> Prediction:
        return self.predict_frames(self._load_frames(path), str(path))

    def predict_batch(self, paths: list[str], batch_size: int = 8) -> list[Prediction]:
        """Batched model forward over many inputs (preprocessing per file)."""
        results: list[Prediction | None] = [None] * len(paths)
        xs, idx = [], []
        for i, p in enumerate(paths):
            x = self.preprocess(self._load_frames(p))
            if x is None:
                results[i] = Prediction(str(p), float("nan"), "unknown", "no_face", 0.0, 0)
                continue
            if x.shape[0] != self.clip_len:  # make clips stackable
                sel = np.linspace(0, x.shape[0] - 1, self.clip_len).round().astype(int)
                x = x[sel]
            xs.append(x), idx.append(i)
        for s in range(0, len(xs), batch_size):
            probs, fps = self._forward(torch.stack(xs[s:s + batch_size]))
            for j, prob in enumerate(probs):
                i = idx[s + j]
                prob = float(prob)
                results[i] = Prediction(str(paths[i]), prob, "fake" if prob >= self.threshold else "real",
                                        self._decide(prob), self._conf(prob), self.clip_len,
                                        fps[j].tolist() if fps is not None else [])
        return results
