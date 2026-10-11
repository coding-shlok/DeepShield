import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="session")
def synthetic_manifest(tmp_path_factory):
    from deepshield.data.synthetic import make_synthetic_dataset

    root = tmp_path_factory.mktemp("synthetic")
    return str(make_synthetic_dataset(root, n_videos=48, n_frames=6, size=64, seed=1))


TINY_MODEL = {
    "arch": "deepshield",
    "backbone": {"name": "resnet18", "pretrained": False, "multiscale": True, "embed_dim": 64, "srm": True},
    "temporal": "bilstm",
    "temporal_kw": {"hidden": 32, "layers": 1, "dropout": 0.0},
    "fusion": {"dim": 32, "heads": 4},
}


@pytest.fixture
def tiny_cfg(tmp_path, synthetic_manifest):
    from deepshield.utils.config import AttrDict

    return AttrDict.wrap({
        "seed": 0,
        "out_dir": str(tmp_path / "run"),
        "data": {"manifests": [synthetic_manifest], "image_size": 48, "clip_len": 3, "batch_size": 4,
                 "eval_batch_size": 8, "num_workers": 0, "augment": {"kyc": {"enabled": True}}},
        "model": dict(TINY_MODEL),
        "loss": {"focal": {"gamma": 2.0, "alpha": [0.5, 0.5]}},
        "train": {"epochs": 1, "lr": 1e-3, "warmup_steps": 2, "max_steps": 6, "log_every": 2, "resume": None,
                  "mixup": {"prob": 0.5, "alpha": 0.4},
                  "adversarial": {"enabled": True, "attack": "pgd", "steps": 1, "prob": 1.0, "weight": 0.5},
                  "gaussian_noise_sigma": 0.1},
    })
