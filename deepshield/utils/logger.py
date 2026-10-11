"""TensorBoard always on; Weights & Biases optional (`logging.wandb: true`)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .distributed import is_main_process


def get_logger(name: str = "deepshield", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO if is_main_process() else logging.WARNING)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    logger.addHandler(h)
    if log_file and is_main_process():
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class ExperimentLogger:
    def __init__(self, out_dir: str | Path, config: dict | None = None, use_wandb: bool = False,
                 project: str = "deepshield", run_name: str | None = None):
        self.out_dir = Path(out_dir)
        self.enabled = is_main_process()
        self.tb = None
        self.wandb = None
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.out_dir / "tb"))
        except Exception as e:  # tensorboard missing
            print(f"[logger] TensorBoard unavailable: {e}")
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb.init(project=project, name=run_name, dir=str(self.out_dir),
                                        config=config.to_dict() if hasattr(config, "to_dict") else config)
            except Exception as e:
                print(f"[logger] W&B disabled: {e}")
        self._jsonl = open(self.out_dir / "metrics.jsonl", "a")

    def log(self, metrics: dict, step: int, prefix: str = "") -> None:
        if not self.enabled:
            return
        flat = {f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if self.tb:
            for k, v in flat.items():
                self.tb.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log(flat, step=step)
        self._jsonl.write(json.dumps({"step": step, **flat}) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        if not self.enabled:
            return
        if self.tb:
            self.tb.close()
        if self.wandb:
            self.wandb.finish()
        self._jsonl.close()
