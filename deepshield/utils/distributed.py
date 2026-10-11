"""Thin helpers around torch.distributed for torchrun-launched DDP (1-4 GPUs)."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed() -> torch.device:
    """Initialise DDP if launched by torchrun; returns the device to use."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        return torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / get_world_size()


def all_gather_cat(t: torch.Tensor) -> torch.Tensor:
    """Gather variable-length 1D/ND tensors from all ranks and concatenate on dim 0."""
    if not is_dist():
        return t
    size = torch.tensor([t.shape[0]], device=t.device)
    sizes = [torch.zeros_like(size) for _ in range(get_world_size())]
    dist.all_gather(sizes, size)
    max_size = int(max(s.item() for s in sizes))
    pad = torch.zeros((max_size - t.shape[0], *t.shape[1:]), dtype=t.dtype, device=t.device)
    padded = torch.cat([t, pad], 0)
    gathered = [torch.zeros_like(padded) for _ in range(get_world_size())]
    dist.all_gather(gathered, padded)
    return torch.cat([g[: int(s.item())] for g, s in zip(gathered, sizes)], 0)


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()
