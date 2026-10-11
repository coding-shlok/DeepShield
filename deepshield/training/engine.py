from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.distributed import all_gather_cat, is_main_process


def autocast(device: torch.device, enabled: bool, dtype: str = "bf16"):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dt = torch.bfloat16 if dtype == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dt)


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW (PRD §6.3: lr 1e-4, wd 1e-5). Backbone can use a lower LR multiplier; no decay on norms/bias."""
    lr, wd = cfg.get("lr", 1e-4), cfg.get("weight_decay", 1e-5)
    bb_mult = cfg.get("backbone_lr_mult", 1.0)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = ".backbone." in f".{n}" or n.startswith("backbone.")
        no_decay = p.ndim <= 1 or n.endswith(".bias") or "pos" in n.split(".")[-1] or "cls" == n.split(".")[-1]
        key = (is_bb, no_decay)
        groups.setdefault(key, []).append(p)
    param_groups = [dict(params=ps, lr=lr * (bb_mult if bb else 1.0), weight_decay=0.0 if nd else wd)
                    for (bb, nd), ps in groups.items()]
    return torch.optim.AdamW(param_groups, lr=lr, betas=tuple(cfg.get("betas", (0.9, 0.999))))


def build_scheduler(opt, total_steps: int, warmup_steps: int = 1000, min_lr_ratio: float = 0.01):
    """Linear warm-up (PRD: 1K steps) then cosine decay."""
    warmup_steps = min(warmup_steps, max(total_steps // 10, 1)) if total_steps < warmup_steps * 2 else warmup_steps

    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def run_inference(model, loader: DataLoader, device, amp: bool = True, transform_batch=None,
                  desc: str = "eval", attack=None) -> dict:
    """Returns numpy arrays: labels, probs (P(fake)), methods, dataset_idx, index, frame_probs (if any).

    `attack(model, x, y) -> x_adv` evaluates under adversarial perturbation (gradients enabled)."""
    model.eval()
    out = {k: [] for k in ("labels", "probs", "methods", "dataset_idx", "index", "frame_probs")}
    it = tqdm(loader, desc=desc, disable=not is_main_process(), leave=False)
    for batch in it:
        x = batch["frames"].to(device, non_blocking=True)
        y = batch["label"].to(device)
        if transform_batch is not None:
            x = transform_batch(x)
        if attack is not None:
            with torch.enable_grad():
                x = attack(model, x, y)
        with autocast(device, amp):
            o = model(x)
        logits = o["logits"] if isinstance(o, dict) else o
        out["probs"].append(logits.float().softmax(-1)[:, 1])
        out["labels"].append(y)
        out["methods"].append(batch["method"].to(device))
        out["dataset_idx"].append(batch["dataset_idx"].to(device))
        out["index"].append(batch["index"].to(device))
        if isinstance(o, dict) and o.get("frame_logits") is not None:
            out["frame_probs"].append(o["frame_logits"].float().softmax(-1)[..., 1])
    res = {}
    for k, v in out.items():
        if not v:
            continue
        t = all_gather_cat(torch.cat(v, 0))
        res[k] = t.cpu().numpy()
    # DistributedSampler pads with duplicates: de-duplicate by dataset index
    if "index" in res:
        _, keep = np.unique(res["index"], return_index=True)
        res = {k: v[keep] for k, v in res.items()}
    return res
