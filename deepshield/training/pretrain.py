"""Stage 1: SimCLR contrastive pretraining on real faces (VGGFace2) + DFFD (PRD §5.1, §6.2 Phase 1).

Produces `backbone.pth` that stage 2 loads via `model.backbone.checkpoint`.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..data.augment import ClipTransform, KYCPerturbation
from ..data.datasets import FaceClipDataset, SimCLRDataset
from ..models.simclr import SimCLR, nt_xent
from ..utils.checkpoint import save_checkpoint, unwrap
from ..utils.config import save_config
from ..utils.distributed import cleanup_distributed, init_distributed, is_dist, is_main_process
from ..utils.logger import ExperimentLogger, get_logger
from ..utils.seed import seed_everything, worker_init_fn
from .engine import autocast, build_optimizer, build_scheduler


def pretrain(cfg) -> str:
    device = init_distributed()
    seed_everything(cfg.get("seed", 42))
    out_dir = Path(cfg.get("out_dir", "runs/pretrain"))
    log = get_logger(log_file=out_dir / "pretrain.log")
    if is_main_process():
        save_config(cfg, out_dir / "config.yaml")
    dc, tc = cfg["data"], cfg["train"]
    size = dc.get("image_size", 224)
    base = FaceClipDataset(dc["manifests"], split=dc.get("split", "train"), transform=None, clip_len=1,
                           max_samples=dc.get("max_train_samples"))
    view_tf = ClipTransform(size=size, train=True, scale=(0.4, 1.0), color_jitter=0.4, grayscale_p=0.2,
                            kyc=KYCPerturbation(random=True))
    ds = SimCLRDataset(base, view_tf)
    sampler = DistributedSampler(ds, shuffle=True) if is_dist() else None
    nw = dc.get("num_workers", 8)
    dl = DataLoader(ds, batch_size=dc.get("batch_size", 128), shuffle=sampler is None, sampler=sampler,
                    num_workers=nw, drop_last=True, pin_memory=True, persistent_workers=nw > 0,
                    worker_init_fn=worker_init_fn)
    model = SimCLR(cfg["model"]["backbone"], **cfg["model"].get("simclr", {})).to(device)
    if is_dist():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    epochs = tc.get("epochs", 100)
    opt = build_optimizer(model, tc)
    sched = build_scheduler(opt, epochs * len(dl), tc.get("warmup_steps", 1000))
    use_amp = tc.get("amp", True) and device.type == "cuda"
    logger = ExperimentLogger(out_dir, cfg, cfg.get("logging", {}).get("wandb", False))
    temp = tc.get("temperature", 0.1)
    step = 0
    log.info(f"SimCLR pretraining on {len(ds)} faces, {len(dl)} it/epoch")
    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        t0, tot = time.time(), 0.0
        for i, b in enumerate(dl):
            v1, v2 = b["view1"].to(device, non_blocking=True), b["view2"].to(device, non_blocking=True)
            with autocast(device, use_amp):
                loss = nt_xent(model(v1), model(v2), temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
            opt.step()
            sched.step()
            step += 1
            tot += float(loss)
            if step % tc.get("log_every", 50) == 0:
                logger.log({"loss": float(loss), "lr": sched.get_last_lr()[0]}, step, "pretrain/")
            if tc.get("max_steps") and step >= tc["max_steps"]:
                break
        log.info(f"epoch {epoch}: nt-xent {tot / max(i + 1, 1):.4f} ({time.time() - t0:.0f}s)")
        if is_main_process():
            save_checkpoint(out_dir / "last.pth", model, epoch=epoch, step=step, config=cfg)
            bb = unwrap(model).backbone
            torch.save({"model": bb.state_dict()}, out_dir / "backbone.pth")
        if tc.get("max_steps") and step >= tc["max_steps"]:
            break
    logger.close()
    cleanup_distributed()
    return str(out_dir / "backbone.pth")
