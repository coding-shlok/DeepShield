from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def plot_roc(curves: dict, path: str | Path, title: str = "ROC") -> None:
    """curves: name -> (y_true, y_prob)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, p) in curves.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import roc_auc_score

        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot([0, 1], [1, 0], ":", color="grey", lw=0.8, label="EER line")
    ax.set_xlabel("False positive rate (real rejected)")
    ax.set_ylabel("True positive rate (fake caught)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path: str | Path, title: str = "Confusion matrix") -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bars(values: dict, path: str | Path, ylabel: str, title: str, target: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(values)), 3.5))
    names = list(values)
    ax.bar(range(len(names)), [values[n] for n in names], color="#4C72B0")
    if target is not None:
        ax.axhline(target, color="#C44E52", ls="--", lw=1, label=f"target {target}")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.0 if max(values.values(), default=1) <= 1 else None)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attack(x, x_adv, saliency, probs_clean, probs_adv, path: str | Path, n: int = 4) -> None:
    """Rows: clean | amplified perturbation | adversarial | input-gradient saliency."""
    n = min(n, x.shape[0])
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 9))
    axes = np.atleast_2d(axes).reshape(4, n)
    for i in range(n):
        delta = (x_adv[i] - x[i])
        amp = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
        rows = [x[i], amp, x_adv[i], saliency[i]]
        titles = [f"clean p(fake)={probs_clean[i]:.2f}", "perturbation (amplified)",
                  f"adv p(fake)={probs_adv[i]:.2f}", "saliency |∇x|"]
        for r in range(4):
            img = rows[r]
            ax = axes[r, i]
            if img.ndim == 3:
                ax.imshow(np.transpose(img, (1, 2, 0)).clip(0, 1))
            else:
                ax.imshow(img, cmap="inferno")
            ax.set_title(titles[r], fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
