import numpy as np
import torch
import torch.nn as nn

from deepshield.evaluation.metrics import compute_eer, compute_metrics
from deepshield.robustness import FeatureSqueezingDetector, Smooth, fgsm, pgd, robustness_confidence
from deepshield.robustness.attacks import apgd_ce, run_autoattack, square_attack


def test_metrics_perfect_and_eer():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["acc"] == 1 and m["auc"] == 1 and m["eer"] == 0 and m["apcer"] == 0 and m["bpcer"] == 0
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    s = rng.normal(y * 1.0, 1.0)
    eer, _ = compute_eer(y, s)
    assert abs(eer - 0.3085) < 0.03  # analytic EER for unit-variance gaussians 1 sigma apart


class Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(3 * 8 * 8, 2)
        torch.manual_seed(0)

    def forward(self, x):
        return self.f(x.flatten(1) - 0.5)


def _data():
    torch.manual_seed(0)
    m = Lin()
    x = torch.rand(32, 3, 8, 8)
    with torch.no_grad():
        y = m(x).argmax(1)  # all correctly classified
    return m, x, y


def test_attacks_reduce_accuracy_and_respect_budget():
    m, x, y = _data()
    eps = 8 / 255
    for adv in [fgsm(m, x, y, eps), pgd(m, x, y, eps, steps=10), apgd_ce(m, x, y, eps, steps=20),
                square_attack(m, x, y, eps, queries=200), run_autoattack(m, x, y, eps, square_queries=100)]:
        assert (adv - x).abs().max() <= eps + 1e-6
        assert adv.min() >= 0 and adv.max() <= 1
        with torch.no_grad():
            acc = (m(adv).argmax(1) == y).float().mean()
        assert acc < 1.0
    # PGD at least as strong as FGSM
    with torch.no_grad():
        a_f = (m(fgsm(m, x, y, eps)).argmax(1) == y).float().mean()
        a_p = (m(pgd(m, x, y, eps, steps=20)).argmax(1) == y).float().mean()
    assert a_p <= a_f + 1e-6


def test_smoothing_and_detection():
    m, x, y = _data()
    pred, radius = Smooth(m, 2, sigma=0.05, batch=50).certify(x[0], n0=20, n=100)
    assert pred in (-1, 0, 1) and radius >= 0
    det = FeatureSqueezingDetector(m)
    s = det.score(x)
    assert s.shape == (32,)
    det.calibrate(s.numpy(), 0.1)
    assert det.detect(x).float().mean() <= 0.2
    rc = robustness_confidence(m, x)
    assert ((rc >= 0) & (rc <= 1)).all()
