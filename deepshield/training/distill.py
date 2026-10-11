"""Knowledge distillation: ResNet-50+BiLSTM teacher -> EfficientNet-B2 student (PRD §6.1, §6.4)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data.datasets import build_dataloader, build_dataset
from ..evaluation.metrics import compute_metrics
from ..losses import FocalLoss, distillation_loss
from ..models.factory import build_model, load_model
from ..utils.checkpoint import save_checkpoint, unwrap
from ..utils.config import save_config
from ..utils.distributed import cleanup_distributed, init_distributed, is_dist, is_main_process
from ..utils.logger import ExperimentLogger, get_logger
from ..utils.seed import seed_everything
from .engine import autocast, build_optimizer, build_scheduler, run_inference


def distill(cfg) -> dict:
    device = init_distributed()
    seed_everything(cfg.get("seed", 42))
    out_dir = Path(cfg.get("out_dir", "runs/distill"))
    log = get_logger(log_file=out_dir / "distill.log")
    if is_main_process():
        save_config(cfg, out_dir / "config.yaml")
    dc, tc, kd = cfg["data"], cfg["train"], cfg["distill"]
    train_dl = build_dataloader(build_dataset(dc, "train"), dc, train=True)
    val_ds = build_dataset(dc, "val")
    val_dl = build_dataloader(val_ds, dc, train=False)

    teacher = load_model(kd["teacher_checkpoint"]).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = build_model(cfg["model"]).to(device)
    model = DDP(student, device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True) if is_dist() else student
    epochs = tc.get("epochs", 10)
    opt = build_optimizer(model, tc)
    sched = build_scheduler(opt, epochs * len(train_dl), tc.get("warmup_steps", 500))
    use_amp = tc.get("amp", True) and device.type == "cuda"
    hard = FocalLoss(**cfg.get("loss", {}).get("focal", {})).to(device)
    logger = ExperimentLogger(out_dir, cfg, cfg.get("logging", {}).get("wandb", False))
    best, step, hist = -1.0, 0, []
    for epoch in range(epochs):
        if hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch)
        model.train()
        for b in train_dl:
            x, y = b["frames"].to(device, non_blocking=True), b["label"].to(device)
            with torch.no_grad(), autocast(device, use_amp):
                t_logits = teacher(x)["logits"]
            with autocast(device, use_amp):
                s_logits = model(x)["logits"]
                loss = distillation_loss(s_logits, t_logits, y, kd.get("temperature", 4.0), kd.get("alpha", 0.7), hard)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
            opt.step()
            sched.step()
            step += 1
            if step % tc.get("log_every", 50) == 0:
                logger.log({"kd_loss": float(loss)}, step, "distill/")
            if tc.get("max_steps") and step >= tc["max_steps"]:
                break
        res = run_inference(unwrap(model), val_dl, device, use_amp, desc="val")
        if is_main_process():
            m = compute_metrics(res["labels"], res["probs"])
            logger.log({k: v for k, v in m.items() if not isinstance(v, list)}, step, "val/")
            log.info(f"epoch {epoch}: student val acc {m['acc']:.4f} auc {m['auc']:.4f}")
            hist.append({"epoch": epoch, "acc": m["acc"], "auc": m["auc"]})
            if np.isfinite(m["auc"]) and m["auc"] > best:
                best = m["auc"]
                thr = m.get("eer_threshold", 0.5)
                save_checkpoint(out_dir / "best.pth", student, epoch=epoch, step=step, best_metric=best, config=cfg,
                                extra={"threshold": float(thr) if np.isfinite(thr) else 0.5})
            save_checkpoint(out_dir / "last.pth", student, epoch=epoch, step=step, config=cfg)
            (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
        if tc.get("max_steps") and step >= tc["max_steps"]:
            break
    logger.close()
    cleanup_distributed()
    return {"best_auc": best, "out_dir": str(out_dir)}
