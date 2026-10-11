import numpy as np
import torch

from deepshield.data.augment import KYC_EVAL_PRESETS, KYCPerturbation, build_eval_transform, build_train_transform


def _clip(T=4, s=96):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (s, s, 3), dtype=np.uint8) for _ in range(T)]


def test_presets_deterministic_and_shape():
    for name in KYC_EVAL_PRESETS:
        tf = build_eval_transform(64, None if name == "clean" else name)
        a, b = tf(_clip()), tf(_clip())
        assert a.shape == (4, 3, 64, 64) and a.dtype == torch.float32
        assert 0 <= a.min() and a.max() <= 1
        assert torch.allclose(a, b), name


def test_train_transform_random():
    tf = build_train_transform({"image_size": 64, "kyc": {"enabled": True}})
    out = tf(_clip())
    assert out.shape == (4, 3, 64, 64)


def test_perturbations_change_image():
    clip = _clip()
    for kw in [dict(resolution_scale=1 / 3), dict(blur_px=5), dict(jpeg_quality=60), dict(brightness=0.7),
               dict(occlusion="mask"), dict(occlusion="glasses"), dict(shadow=0.4), dict(noise_sigma=5.0)]:
        base = dict(resolution_scale=None, blur_px=None, jitter_px=None, jpeg_quality=None, brightness=None,
                    shadow=None, occlusion=None, noise_sigma=None, random=False, seed=0)
        out = KYCPerturbation(**(base | kw))(clip)
        assert any((o != c).any() for o, c in zip(out, clip)), kw
