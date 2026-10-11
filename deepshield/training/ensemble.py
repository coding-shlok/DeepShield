"""Fit the adaptive voting ensemble on cached validation logits (PRD Contribution 3)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..data.datasets import build_dataloader, build_dataset
from ..evaluation.metrics import compute_metrics
from ..models.ensemble import AdaptiveEnsemble, fit_ensemble
from ..models.factory import load_model
from ..utils.checkpoint import save_checkpoint
from ..utils.distributed import init_distributed


@torch.no_grad()
def _member_logits(ens, dl, device):
    L, Y = [], []
    for b in dl:
        L.append(ens.member_logits(b["frames"].to(device)).cpu())
        Y.append(b["label"])
    return torch.cat(L), torch.cat(Y)


def fit_ensemble_cmd(cfg) -> dict:
    device = init_distributed()
    ec = cfg["ensemble"]
    out = Path(cfg.get("out_dir", "runs/ensemble"))
    out.mkdir(parents=True, exist_ok=True)
    members = {n: load_model(m["checkpoint"]) for n, m in ec["members"].items()}
    ens = AdaptiveEnsemble(members, mode=ec.get("mode", "gated")).to(device)
    dc = cfg["data"]
    val_dl = build_dataloader(build_dataset(dc, "val", train=False), dc, train=False)
    test_dl = build_dataloader(build_dataset(dc, "test", train=False), dc, train=False)
    lv, yv = _member_logits(ens, val_dl, device)
    lt, yt = _member_logits(ens, test_dl, device)
    ens = ens.cpu()
    fit_ensemble(ens, lv, yv, epochs=ec.get("epochs", 300), lr=ec.get("lr", 1e-2))
    report = {}
    with torch.no_grad():
        for i, name in enumerate(members):
            report[name] = compute_metrics(yt.numpy(), lt[:, i].softmax(-1)[:, 1].numpy())
        for mode in ("mean", "confidence", ens.mode):
            saved = ens.mode
            ens.mode = mode
            report[f"ensemble_{mode}"] = compute_metrics(yt.numpy(), ens.combine(lt).exp()[:, 1].numpy())
            ens.mode = saved
    model_cfg = {"arch": "ensemble", "mode": ens.mode,
                 "members": {n: {"checkpoint": m["checkpoint"]} for n, m in ec["members"].items()}}
    save_checkpoint(out / "ensemble.pth", ens, config={**cfg.to_dict(), "model": model_cfg}
                    if hasattr(cfg, "to_dict") else {**cfg, "model": model_cfg})
    clean = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in report.items()}
    (out / "ensemble_report.json").write_text(json.dumps(clean, indent=2))
    for k, v in clean.items():
        print(f"{k:24s} acc={v['acc']:.4f} auc={v['auc']:.4f} eer={v['eer']:.4f}")
    return clean
