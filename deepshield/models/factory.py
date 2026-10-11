from __future__ import annotations

import copy

import torch.nn as nn

from ..utils.checkpoint import load_checkpoint


def build_model(cfg: dict) -> nn.Module:
    cfg = copy.deepcopy(dict(cfg))
    arch = cfg.pop("arch", "deepshield")
    if arch == "deepshield":
        from .detector import DeepShieldNet

        return DeepShieldNet(**cfg)
    if arch == "video3d":
        from .video3d import Video3DNet

        return Video3DNet(**cfg)
    if arch == "ensemble":
        from .ensemble import AdaptiveEnsemble

        members = {}
        for name, m in cfg.pop("members").items():
            members[name] = load_model(m["checkpoint"], m.get("model"))
        return AdaptiveEnsemble(members, **cfg)
    raise ValueError(f"unknown arch {arch}")


def load_model(checkpoint: str, model_cfg: dict | None = None, map_location="cpu", strict: bool = True) -> nn.Module:
    """Rebuild a model from the config stored in its checkpoint and load weights."""
    import torch

    state = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if model_cfg is None:
        model_cfg = (state.get("config") or {}).get("model")
        if model_cfg is None:
            raise ValueError(f"{checkpoint} has no stored config; pass model_cfg")
    model_cfg = copy.deepcopy(dict(model_cfg))
    # weights come from the checkpoint – avoid downloading ImageNet weights again
    if "backbone" in model_cfg:
        model_cfg["backbone"] = dict(model_cfg["backbone"], pretrained=False, checkpoint=None)
    if model_cfg.get("arch") == "video3d":
        model_cfg["pretrained"] = False
    model = build_model(model_cfg)
    load_checkpoint(checkpoint, model, map_location=map_location, strict=strict)
    # decision threshold calibrated on validation (EER point) during training
    model.calibrated_threshold = float(state.get("threshold", 0.5))
    model.data_config = dict((state.get("config") or {}).get("data") or {})
    return model.eval()
