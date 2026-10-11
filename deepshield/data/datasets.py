"""Datasets over pre-extracted face crops.

Processed layout (written by `deepshield prepare`):
    <root>/manifest.csv
    <root>/faces/<sample_id>/000.jpg 001.jpg ...     (video: N face crops)
    <root>/faces/<sample_id>.jpg                       (image dataset: single crop)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from ..utils.distributed import get_rank, get_world_size, is_dist
from ..utils.seed import worker_init_fn
from .augment import ClipTransform, build_eval_transform, build_train_transform
from .constants import method_index
from .manifest import read_manifest

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _load_rgb(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


class FaceClipDataset(Dataset):
    """Returns dict(frames=[T,3,S,S] float [0,1], label, method, dataset_idx, index).

    clip_len=1 gives single-frame (image-level) samples. For training, a random temporal
    window is chosen; for evaluation, frames are sampled uniformly over the whole video.
    """

    def __init__(self, manifests: Sequence[str | Path] | str | Path, split: str | None,
                 transform: ClipTransform, clip_len: int = 16, train: bool = True,
                 frame_stride: int | Sequence[int] = 1, max_samples: int | None = None,
                 datasets: Sequence[str] | None = None, seed: int = 0):
        if isinstance(manifests, (str, Path)):
            manifests = [manifests]
        frames = []
        for m in manifests:
            df = read_manifest(m)
            df["abs_path"] = [str(Path(df.attrs["root"]) / p) for p in df["path"]]
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        if split:
            df = df[df["split"] == split]
        if datasets:
            df = df[df["dataset"].isin(datasets)]
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed)
        self.df = df.reset_index(drop=True)
        self.dataset_names = sorted(self.df["dataset"].unique().tolist())
        self.transform = transform
        self.clip_len = clip_len
        self.train = train
        self.frame_stride = [frame_stride] if isinstance(frame_stride, int) else list(frame_stride)
        self._frame_cache: dict[int, list[str]] = {}

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy().astype(int)

    def _frame_paths(self, idx: int) -> list[str]:
        if idx not in self._frame_cache:
            p = Path(self.df.at[idx, "abs_path"])
            if p.is_dir():
                files = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
            else:
                files = [str(p)]
            if not files:
                raise RuntimeError(f"No frames in {p}")
            self._frame_cache[idx] = files
        return self._frame_cache[idx]

    def _select(self, files: list[str]) -> list[str]:
        n, T = len(files), self.clip_len
        if self.train:
            stride = random.choice(self.frame_stride)
            span = (T - 1) * stride + 1
            if n >= span:
                start = random.randint(0, n - span)
                idx = list(range(start, start + span, stride))
            else:
                idx = np.linspace(0, n - 1, T).round().astype(int).tolist()
        else:
            idx = np.linspace(0, n - 1, T).round().astype(int).tolist() if n > 1 else [0] * T
        return [files[i] for i in idx]

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        try:
            clip = [_load_rgb(f) for f in self._select(self._frame_paths(idx))]
        except Exception as e:  # corrupt file: fall back to another sample instead of crashing a long job
            print(f"[dataset] failed {row['abs_path']}: {e}")
            return self[random.randint(0, len(self) - 1)]
        h, w = clip[0].shape[:2]
        clip = [c if c.shape[:2] == (h, w) else cv2.resize(c, (w, h)) for c in clip]
        label = int(row["label"])
        return {
            "frames": self.transform(clip),
            "label": torch.tensor(label, dtype=torch.long),
            "method": torch.tensor(method_index(row["method"], label), dtype=torch.long),
            "dataset_idx": torch.tensor(self.dataset_names.index(row["dataset"]), dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class SimCLRDataset(Dataset):
    """Two augmented views of the same face crop for contrastive pretraining (PRD §6.2 Phase 1)."""

    def __init__(self, base: FaceClipDataset, transform: ClipTransform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        files = self.base._frame_paths(idx)
        im = _load_rgb(random.choice(files))
        v1 = self.transform([im])[0]
        v2 = self.transform([im])[0]
        return {"view1": v1, "view2": v2}


# ----------------------------------------------------------------------------- samplers
class DistributedWeightedSampler(Sampler):
    """WeightedRandomSampler that also shards across DDP ranks."""

    def __init__(self, weights: np.ndarray, num_samples: int | None = None, seed: int = 0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.world, self.rank = get_world_size(), get_rank()
        total = num_samples or len(weights)
        self.num_samples = int(np.ceil(total / self.world))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.num_samples * self.world, replacement=True, generator=g)
        return iter(idx[self.rank::self.world].tolist())

    def __len__(self):
        return self.num_samples


def sample_weights(df: pd.DataFrame, balance_classes: bool = True,
                   dataset_weights: dict | None = None) -> np.ndarray:
    """Per-sample weights: class balancing within each dataset x optional dataset mixing weights."""
    w = np.ones(len(df), dtype=np.float64)
    for ds, sub in df.groupby("dataset"):
        idx = sub.index.to_numpy()
        if balance_classes:
            counts = sub["label"].value_counts()
            w[idx] = sub["label"].map(lambda l: 1.0 / counts[l]).to_numpy()
        else:
            w[idx] = 1.0
        w[idx] /= w[idx].sum()
        w[idx] *= (dataset_weights or {}).get(ds, 1.0)
    return w / w.sum()


def subsample_ratio(df: pd.DataFrame, real_frac: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """Subsample to a target Real:Fake ratio (e.g. 80:20 KYC prior, PRD §5.3/§8.3)."""
    real, fake = df[df.label == 0], df[df.label == 1]
    # largest subset with exact ratio
    n_fake = min(len(fake), max(1, int(len(real) * (1 - real_frac) / real_frac)))
    n_real = min(len(real), max(1, int(round(n_fake * real_frac / (1 - real_frac)))))
    return pd.concat([real.sample(n=n_real, random_state=seed), fake.sample(n=n_fake, random_state=seed)])


# ----------------------------------------------------------------------------- builders
def build_dataset(data_cfg: dict, split: str, train: bool | None = None, preset: str | None = None,
                  manifests: Sequence[str] | None = None) -> FaceClipDataset:
    train = (split == "train") if train is None else train
    size = data_cfg.get("image_size", 224)
    tf = build_train_transform(data_cfg.get("augment", {}) | {"image_size": size}) if train \
        else build_eval_transform(size, preset)
    return FaceClipDataset(
        manifests or data_cfg["manifests"], split=split, transform=tf,
        clip_len=data_cfg.get("clip_len", 16), train=train,
        frame_stride=data_cfg.get("frame_stride", 1),
        max_samples=data_cfg.get(f"max_{split}_samples"),
        datasets=data_cfg.get("datasets"), seed=data_cfg.get("seed", 0),
    )


def build_dataloader(ds: Dataset, data_cfg: dict, train: bool, sampler: Sampler | None = None) -> DataLoader:
    if sampler is None:
        if train and isinstance(ds, FaceClipDataset) and data_cfg.get("balanced_sampling", True):
            w = sample_weights(ds.df, True, data_cfg.get("dataset_weights"))
            sampler = DistributedWeightedSampler(w, num_samples=data_cfg.get("samples_per_epoch") or len(ds),
                                                 seed=data_cfg.get("seed", 0))
        elif is_dist():
            sampler = DistributedSampler(ds, shuffle=train, drop_last=False)
    nw = data_cfg.get("num_workers", 8)
    return DataLoader(
        ds, batch_size=data_cfg.get("batch_size", 16), sampler=sampler,
        shuffle=(train and sampler is None), num_workers=nw, pin_memory=torch.cuda.is_available(),
        drop_last=train, persistent_workers=nw > 0, worker_init_fn=worker_init_fn,
        prefetch_factor=data_cfg.get("prefetch_factor", 2) if nw > 0 else None,
    )
