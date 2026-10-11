"""Face detectors: RetinaFace (insightface) -> MTCNN (facenet-pytorch) -> center-crop fallback."""
from __future__ import annotations

import numpy as np


class FaceDetector:
    name = "base"

    def detect(self, rgb: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Returns list of (bbox[x1,y1,x2,y2], score)."""
        raise NotImplementedError


class RetinaFaceDetector(FaceDetector):
    name = "retinaface"

    def __init__(self, device: str = "cuda", det_size: int = 640, model: str = "buffalo_l"):
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") \
            else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model, allowed_modules=["detection"], providers=providers)
        ctx = 0 if device.startswith("cuda") else -1
        self.app.prepare(ctx_id=ctx, det_size=(det_size, det_size))

    def detect(self, rgb):
        faces = self.app.get(rgb[:, :, ::-1])  # insightface expects BGR
        return [(f.bbox.astype(np.float32), float(f.det_score)) for f in faces]


class MTCNNDetector(FaceDetector):
    name = "mtcnn"

    def __init__(self, device: str = "cuda"):
        import torch
        from facenet_pytorch import MTCNN

        dev = device if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(keep_all=True, device=dev, post_process=False)

    def detect(self, rgb):
        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None:
            return []
        return [(b.astype(np.float32), float(p)) for b, p in zip(boxes, probs)]


class CenterCropDetector(FaceDetector):
    """No detection: whole frame (pre-cropped datasets like WildDeepfake / DeepfakeBench)."""

    name = "none"

    def detect(self, rgb):
        h, w = rgb.shape[:2]
        return [(np.array([0, 0, w, h], np.float32), 1.0)]


def build_detector(name: str = "retinaface", device: str = "cuda") -> FaceDetector:
    order = {"retinaface": ["retinaface", "mtcnn"], "mtcnn": ["mtcnn"], "none": ["none"]}[name]
    for n in order:
        try:
            if n == "retinaface":
                return RetinaFaceDetector(device)
            if n == "mtcnn":
                return MTCNNDetector(device)
            return CenterCropDetector()
        except Exception as e:  # missing optional dependency
            print(f"[face_detect] {n} unavailable ({e}); trying fallback")
    print("[face_detect] WARNING: no detector available, using full-frame crops")
    return CenterCropDetector()
