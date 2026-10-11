import torch

from deepshield.losses import FocalLoss, MultiTaskLoss, contrastive_margin_loss, temporal_consistency_loss
from deepshield.models import AdaptiveEnsemble, DeepShieldNet, fit_ensemble
from deepshield.models.factory import build_model

from .conftest import TINY_MODEL


def test_deepshield_forward_clip_and_image():
    m = build_model(TINY_MODEL).eval()
    out = m(torch.rand(2, 3, 3, 48, 48))
    assert out["logits"].shape == (2, 2)
    assert out["frame_logits"].shape == (2, 3, 2)
    assert out["method_logits"].shape[0] == 2
    assert out["temporal_score"].shape == (2,)
    assert m(torch.rand(2, 3, 48, 48))["logits"].shape == (2, 2)  # single selfie


def test_variants_forward():
    for cfg in [
        {"backbone": {"name": "efficientnet_b0", "pretrained": False, "embed_dim": 32}, "temporal": "none",
         "fusion": {"dim": 32, "heads": 4}},
        {"backbone": {"name": "vit_tiny_patch16_224", "pretrained": False, "multiscale": False, "img_size": 64},
         "temporal": "tconv", "temporal_kw": {"hidden": 32}, "fusion": {"dim": 32, "heads": 4}},
    ]:
        m = DeepShieldNet(**cfg).eval()
        assert m(torch.rand(1, 2, 3, 64, 64))["logits"].shape == (1, 2)
    v = build_model({"arch": "video3d", "arch_name": "r3d_18", "pretrained": False}).eval()
    assert v(torch.rand(1, 4, 3, 32, 32))["logits"].shape == (1, 2)


def test_losses():
    logits = torch.randn(6, 2, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 1, 0])
    fl = FocalLoss(alpha=(0.25, 0.75))(logits, y)
    fl.backward()
    assert fl.item() > 0
    # gamma=0, alpha=None reduces to cross entropy
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.isclose(FocalLoss(gamma=0, alpha=None)(logits, y), ce)
    assert temporal_consistency_loss(torch.zeros(2, 4, 2)).item() == 0
    emb = torch.randn(6, 8)
    assert contrastive_margin_loss(emb, y).item() >= 0
    m = build_model(TINY_MODEL)
    out = m(torch.rand(4, 3, 3, 48, 48))
    total, parts = MultiTaskLoss()(out, torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 2]))
    total.backward()
    assert set(parts) >= {"focal", "frame", "temporal", "method", "temporal_score", "contrastive"}


def test_ensemble_fit_improves():
    torch.manual_seed(0)
    y = torch.randint(0, 2, (200,))
    sgn = (y * 2 - 1).float()
    good = torch.stack([-2 * sgn, 2 * sgn], 1) + torch.randn(200, 2) * 0.5
    noise = torch.randn(200, 2) * 2
    ens = AdaptiveEnsemble({"a": torch.nn.Identity(), "b": torch.nn.Identity()}, mode="weighted")
    logits = torch.stack([good, noise], 1)
    fit_ensemble(ens, logits, y, epochs=200)
    assert ens.weights.softmax(0)[0] > 0.6
