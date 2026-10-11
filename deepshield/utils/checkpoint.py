from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str | Path, model: nn.Module, optimizer=None, scheduler=None,
                    scaler=None, epoch: int = 0, step: int = 0, best_metric: float | None = None,
                    config: dict | None = None, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model": unwrap(model).state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra:
        state.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, model: nn.Module | None = None, optimizer=None, scheduler=None,
                    scaler=None, map_location="cpu", strict: bool = True) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        sd = state.get("model", state)
        missing, unexpected = unwrap(model).load_state_dict(sd, strict=strict)
        if not strict and (missing or unexpected):
            print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state
