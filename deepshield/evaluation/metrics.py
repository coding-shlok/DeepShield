"""Evaluation metrics (PRD §8). Positive class = fake (attack).

KYC terminology (ISO/IEC 30107-3):
  APCER / FAR : fraction of fakes accepted as real  (fraud gets through)
  BPCER / FRR : fraction of reals rejected as fake  (genuine customer rejected)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, roc_auc_score, roc_curve)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where FPR == FNR (linear interpolation on the ROC)."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    # interpolate between i and neighbour for a smoother estimate
    j = i + 1 if i + 1 < len(fpr) and (fnr[i] - fpr[i]) > 0 else max(i - 1, 0)
    d1, d2 = fnr[i] - fpr[i], fnr[j] - fpr[j]
    if d1 != d2 and np.sign(d1) != np.sign(d2):
        t = d1 / (d1 - d2)
        eer = fpr[i] + t * (fpr[j] - fpr[i])
        th = thr[i] + t * (thr[j] - thr[i]) if np.isfinite(thr[i]) and np.isfinite(thr[j]) else thr[i]
    else:
        eer, th = (fpr[i] + fnr[i]) / 2, thr[i]
    return float(eer), float(th)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    m = {
        "n": int(len(y_true)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "acc": float((tp + tn) / max(len(y_true), 1)),
        "balanced_acc": float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
        "auc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "ap": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),  # "balanced F1"
        "precision_fake": float(tp / max(tp + fp, 1)),
        "recall_fake": float(tp / max(tp + fn, 1)),
        "apcer": float(fn / max(tp + fn, 1)),   # fake accepted
        "bpcer": float(fp / max(tn + fp, 1)),   # real rejected
        "threshold": float(threshold),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    m["eer"], m["eer_threshold"] = compute_eer(y_true, y_prob)
    m["hter"] = 0.5 * (m["apcer"] + m["bpcer"])
    if both:  # operating points relevant for KYC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        for target in (0.01, 0.05):
            k = np.searchsorted(fpr, target, side="right") - 1
            m[f"tpr@fpr{int(target * 100)}"] = float(tpr[max(k, 0)])
    return m


def generalization_gap(acc_benchmark: float, acc_unseen: float) -> float:
    """PRD §8.2: gap = Acc_benchmark - Acc_unseen (target <= 0.10)."""
    return float(acc_benchmark - acc_unseen)


def bootstrap_ci(y_true, y_prob, metric: str = "auc", n: int = 1000, alpha: float = 0.05, seed: int = 0,
                 threshold: float = 0.5) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(compute_metrics(y_true[idx], y_prob[idx], threshold)[metric])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


PRD_TARGETS = {
    # dataset -> metric -> (op, value)
    "dfdc": {"acc": (">=", 0.95), "auc": (">=", 0.98), "eer": ("<=", 0.05)},
    "faceforensics": {"acc": (">=", 0.90), "auc": (">=", 0.98)},
    "celebdf": {"acc": (">=", 0.88)},
    "wilddeepfake": {"acc": (">=", 0.88), "auc": (">=", 0.92), "eer": ("<=", 0.08)},
    "kyc_combined": {"acc": (">=", 0.85)},
    "adversarial": {"acc": (">=", 0.85)},
    "imbalance_80_20": {"f1_macro": (">=", 0.92), "bpcer": ("<", 0.03)},
    "generalization_gap": {"gap": ("<=", 0.10)},
    "latency": {"gpu_ms": ("<", 100), "cpu_ms": ("<", 300)},
}


def check_target(value: float, op: str, target: float) -> bool:
    return {">=": value >= target, "<=": value <= target, "<": value < target, ">": value > target}[op]
