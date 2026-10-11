"""Spatial feature extractors (PRD §4.2 'Spatial Feature Extraction').

* any timm model (resnet50, efficientnet_b2, vit_base_patch16_224, ...)
* optional multi-scale feature pyramid: pooled features of the last `levels` stages are
  projected and concatenated, capturing artefacts at several frequency bands / scales
* optional SRM high-pass residual stream (noise-domain cues of blending / up-sampling)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _srm_kernels() -> torch.Tensor:
    k1 = torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4.0
    k2 = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]]) / 12.0
    k3 = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2.0
    k = torch.stack([k1, k2, k3]).float()  # 3,5,5
    return k[:, None].repeat(1, 3, 1, 1) / 3.0  # 3 out, 3 in


class SRMStream(nn.Module):
    """Fixed SRM high-pass filters + light CNN -> `out_dim` vector."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.register_buffer("srm", _srm_kernels())
        c = 32
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, 2 * c, 3, 2, 1, bias=False), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1, bias=False), nn.BatchNorm2d(4 * c), nn.ReLU(inplace=True),
            nn.Conv2d(4 * c, out_dim, 3, 2, 1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):
        res = F.conv2d(x, self.srm, padding=2).clamp(-3, 3)
        return self.net(res)


class SpatialBackbone(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True, multiscale: bool = True,
                 levels: int = 3, embed_dim: int | None = None, srm: bool = False, srm_dim: int = 256,
                 drop_path: float = 0.0, img_size: int = 224, checkpoint: str | None = None,
                 grad_checkpointing: bool = False):
        super().__init__()
        self.name = name
        kw = dict(pretrained=pretrained)
        if drop_path:
            kw["drop_path_rate"] = drop_path
        self.is_vit = any(k in name for k in ("vit", "deit", "swin", "beit", "eva"))
        self.multiscale = multiscale and not self.is_vit
        if self.is_vit:
            kw["img_size"] = img_size
        if self.multiscale:
            self.body = timm.create_model(name, features_only=True, out_indices=tuple(range(-levels, 0)), **kw)
            chs = self.body.feature_info.channels()
            final = chs[-1]
            self.laterals = nn.ModuleList([nn.Conv2d(c, max(final // 4, 64), 1) for c in chs[:-1]])
            feat_dim = final + sum(max(final // 4, 64) for _ in chs[:-1])
        else:
            self.body = timm.create_model(name, num_classes=0, **kw)
            feat_dim = self.body.num_features
        if grad_checkpointing and hasattr(self.body, "set_grad_checkpointing"):
            self.body.set_grad_checkpointing(True)
        self.srm = SRMStream(srm_dim) if srm else None
        if self.srm is not None:
            feat_dim += srm_dim
        self.out_dim = embed_dim or feat_dim
        self.proj = nn.Identity() if embed_dim is None or embed_dim == feat_dim else nn.Sequential(
            nn.Linear(feat_dim, embed_dim), nn.LayerNorm(embed_dim))
        if checkpoint:
            self.load_pretrained(checkpoint)

    def load_pretrained(self, path: str) -> None:
        """Load SimCLR/VGGFace2-pretrained weights (keys prefixed `backbone.` are stripped)."""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        sd = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[backbone] loaded {path}: missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N,3,H,W] already normalised -> [N, out_dim]."""
        if self.multiscale:
            feats = self.body(x)
            if feats[0].dim() == 4 and feats[0].shape[1] != self.body.feature_info.channels()[0]:
                feats = [f.permute(0, 3, 1, 2) for f in feats]  # NHWC (convnext-style) safety
            pooled = [F.adaptive_avg_pool2d(lat(f), 1).flatten(1) for lat, f in zip(self.laterals, feats[:-1])]
            pooled.append(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))
            z = torch.cat(pooled, 1)
        else:
            z = self.body(x)
        if self.srm is not None:
            z = torch.cat([z, self.srm(x)], 1)
        return self.proj(z)
