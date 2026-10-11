"""Supervised training loop for detection models (PRD §6.2 Phases 2 & 3).

    torchrun --nproc_per_node=1 -m deepshield.cli train --config configs/experiments/stage2_resnet_bilstm.yaml
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data.datasets import build_dataloader, build_dataset
from ..evaluation.metrics import compute_metrics
from ..losses import MultiTaskLoss
from ..models.factory import build_model
from ..robustness.attacks import fgsm, pgd
from ..utils.checkpoint import load_checkpoint, save_checkpoint, unwrap
from ..utils.config import save_config
from ..utils.distributed import (all_reduce_mean, barrier, cleanup_distributed, init_distributed,
                                 is_dist, is_main_process)
from ..utils.logger import ExperimentLogger, get_logger
from ..utils.seed import seed_everything
from .engine import autocast, build_optimizer, build_scheduler, run_inference


def _mixup(x, y_onehot, method, alpha):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    x = lam * x + (1 - lam) * x[perm]
    y = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return x, y, method[perm], lam


def train(cfg) -> dict:
    device = init_distributed()
    seed_everything(cfg.get("seed", 42), cfg.get("deterministic", False))
    out_dir = Path(cfg.get("out_dir", "runs/default"))
    log = get_logger(log_file=out_dir / "train.log")
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")
    tc, dc = cfg["train"], cfg["data"]

    # ---------------------------------------------------------------- data
    train_ds = build_dataset(dc, "train")
    val_ds = build_dataset(dc, "val")
    train_dl = build_dataloader(train_ds, dc, train=True)
    val_dl = build_dataloader(val_ds, dc | {"batch_size": dc.get("eval_batch_size", dc.get("batch_size", 16))},
                              train=False)
    log.info(f"train={len(train_ds)} val={len(val_ds)} datasets={train_ds.dataset_names}")

    # ---------------------------------------------------------------- model
    model = build_model(cfg["model"]).to(device)
    if tc.get("init_checkpoint"):
        load_checkpoint(tc["init_checkpoint"], model, strict=False)
        log.info(f"initialised from {tc['init_checkpoint']}")
    if tc.get("channels_last", True) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = MultiTaskLoss(**cfg.get("loss", {})).to(device)

    epochs = tc.get("epochs", 20)
    accum = tc.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(train_dl) / accum)
    opt = build_optimizer(model, tc)
    freeze_epochs = tc.get("freeze_backbone_epochs", 0)
    bb_params = list(unwrap(model).backbone.parameters()) if hasattr(unwrap(model), "backbone") else []
    sched = build_scheduler(opt, epochs * steps_per_epoch, tc.get("warmup_steps", 1000), tc.get("min_lr_ratio", 0.01))
    use_amp = tc.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not torch.cuda.is_bf16_supported())

    start_epoch, step, best = 0, 0, -float("inf")
    resume = tc.get("resume")
    if resume == "auto" and (out_dir / "last.pth").exists():
        resume = str(out_dir / "last.pth")
    if resume and resume != "auto" and Path(resume).exists():
        st = load_checkpoint(resume, model, opt, sched, scaler, map_location=device)
        start_epoch, step, best = st["epoch"] + 1, st["step"], st.get("best_metric") or best
        log.info(f"resumed from {resume} at epoch {start_epoch}")

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=tc.get("find_unused_parameters", True)) if is_dist() else model
    if tc.get("compile", False):
        ddp_model = torch.compile(ddp_model)

    logger = ExperimentLogger(out_dir, cfg, use_wandb=cfg.get("logging", {}).get("wandb", False),
                              project=cfg.get("logging", {}).get("project", "deepshield"),
                              run_name=cfg.get("name"))
    mix = tc.get("mixup", {}) or {}
    adv = tc.get("adversarial", {}) or {}
    noise_sigma = tc.get("gaussian_noise_sigma", 0.0)
    select = tc.get("select_metric", "auc")
    patience, bad_epochs = tc.get("early_stop_patience", 0), 0
    history = []

    for epoch in range(start_epoch, epochs):
        if hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch)
        frozen = epoch < freeze_epochs  # warm up heads first; grads dropped so DDP stays consistent
        if frozen:
            log.info(f"epoch {epoch}: backbone frozen")
        ddp_model.train()
        t0, run = time.time(), {}
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_dl):
            x = batch["frames"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            meth = batch["method"].to(device, non_blocking=True)
            y_soft, meth_b, lam = F.one_hot(y, 2).float(), None, 1.0

            # ---- adversarial training (PRD Phase 3/4): craft with frozen BN on the unwrapped model
            adv_on = adv.get("enabled", False) and epoch >= adv.get("start_epoch", 0) \
                and np.random.rand() < adv.get("prob", 0.5)
            if adv_on:
                base = unwrap(model)
                base.eval()
                eps = adv.get("eps", 8 / 255)
                if adv.get("attack", "pgd") == "fgsm":
                    x_adv = fgsm(base, x, y, eps)
                else:
                    x_adv = pgd(base, x, y, eps, steps=adv.get("steps", 3))
                ddp_model.train()
            if noise_sigma > 0 and np.random.rand() < tc.get("gaussian_noise_prob", 0.5):
                x = (x + noise_sigma * torch.randn_like(x)).clamp(0, 1)
            if mix.get("prob", 0) > 0 and np.random.rand() < mix["prob"] and not adv_on:
                x, y_soft, meth_b, lam = _mixup(x, y_soft, meth, mix.get("alpha", 0.4))

            sync = (i + 1) % accum == 0 or (i + 1) == len(train_dl)
            ctx = ddp_model.no_sync() if (is_dist() and not sync) else torch.enable_grad()
            with ctx:
                with autocast(device, use_amp):
                    out = ddp_model(x)
                    loss, parts = criterion(out, y_soft, meth, meth_b, lam)
                    if adv_on:
                        out_adv = ddp_model(x_adv)
                        loss_adv, _ = criterion(out_adv, F.one_hot(y, 2).float(), meth)
                        w = adv.get("weight", 0.5)
                        loss = (1 - w) * loss + w * loss_adv
                        parts["adv"] = float(loss_adv.detach())
                scaler.scale(loss / accum).backward()
            if sync:
                if frozen:
                    for p in bb_params:
                        p.grad = None
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                parts["grad_norm"] = float(gn)
            parts["loss"] = float(loss.detach())
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            if sync and step % tc.get("log_every", 50) == 0:
                logger.log({**parts, "lr": sched.get_last_lr()[0]}, step, "train/")
                log.info(f"ep {epoch} it {i + 1}/{len(train_dl)} step {step} loss {parts['loss']:.4f} "
                         f"lr {sched.get_last_lr()[0]:.2e}")
            if tc.get("max_steps") and step >= tc["max_steps"]:
                break
        n = max(i + 1, 1)
        train_stats = {k: v / n for k, v in run.items()}
        train_stats["epoch_time_s"] = time.time() - t0

        # ---------------------------------------------------------------- validation
        res = run_inference(unwrap(ddp_model) if not tc.get("compile") else ddp_model, val_dl, device, use_amp,
                            desc=f"val {epoch}")
        metrics = compute_metrics(res["labels"], res["probs"]) if is_main_process() else {}
        if is_main_process():
            # per-dataset breakdown
            for di, name in enumerate(val_ds.dataset_names):
                msk = res["dataset_idx"] == di
                if msk.sum() > 0 and len(np.unique(res["labels"][msk])) == 2:
                    sub = compute_metrics(res["labels"][msk], res["probs"][msk])
                    metrics[f"{name}/auc"], metrics[f"{name}/acc"] = sub["auc"], sub["acc"]
            logger.log({k: v for k, v in metrics.items() if not isinstance(v, list)}, step, "val/")
            score = metrics.get(select, float("nan"))
            score = -score if select in ("eer", "hter", "apcer", "bpcer") else score
            improved = np.isfinite(score) and score > best
            if improved:
                best, bad_epochs = score, 0
                thr = metrics.get("eer_threshold")
                save_checkpoint(out_dir / "best.pth", model, epoch=epoch, step=step, best_metric=best, config=cfg,
                                extra={"threshold": float(thr) if thr is not None and np.isfinite(thr) else 0.5})
            else:
                bad_epochs += 1
            save_checkpoint(out_dir / "last.pth", model, opt, sched, scaler, epoch, step, best, cfg)
            log.info(f"epoch {epoch}: train_loss {train_stats.get('loss', 0):.4f} | val acc {metrics['acc']:.4f} "
                     f"auc {metrics['auc']:.4f} eer {metrics['eer']:.4f} | best {select} {abs(best):.4f}")
            history.append({"epoch": epoch, "train": train_stats,
                            "val": {k: v for k, v in metrics.items() if not isinstance(v, list)}})
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        stop = torch.tensor(int(patience > 0 and bad_epochs >= patience), device=device)
        if is_dist():
            stop = all_reduce_mean(stop.float())
        barrier()
        if stop.item() > 0 or (tc.get("max_steps") and step >= tc["max_steps"]):
            log.info("early stopping")
            break

    logger.close()
    cleanup_distributed()
    return {"best": abs(best) if np.isfinite(best) else None, "out_dir": str(out_dir), "history": history}
