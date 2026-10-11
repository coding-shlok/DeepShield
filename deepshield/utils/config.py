"""YAML config loading with `_base_` inheritance and dotted CLI overrides.

Example:
    cfg = load_config("configs/experiments/train_resnet_bilstm.yaml",
                      ["train.epochs=5", "data.batch_size=8"])
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


class AttrDict(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return AttrDict(copy.deepcopy(dict(self), memo))

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict) and not isinstance(obj, AttrDict):
            return AttrDict({k: AttrDict.wrap(v) for k, v in obj.items()})
        if isinstance(obj, AttrDict):
            for k, v in obj.items():
                obj[k] = AttrDict.wrap(v)
            return obj
        if isinstance(obj, list):
            return [AttrDict.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and v.get("_replace_"):  # `_replace_: true` -> do not inherit this section
            out[k] = {kk: copy.deepcopy(vv) for kk, vv in v.items() if kk != "_replace_"}
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = _deep_merge(merged, _load_yaml((path.parent / b).resolve()))
    return _deep_merge(merged, data)


def _parse_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply `a.b.c=value` overrides (value parsed as YAML)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | os.PathLike, overrides: Iterable[str] | None = None) -> AttrDict:
    cfg = _load_yaml(Path(path).resolve())
    cfg = apply_overrides(cfg, overrides)
    return AttrDict.wrap(cfg)


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, AttrDict) else cfg
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
