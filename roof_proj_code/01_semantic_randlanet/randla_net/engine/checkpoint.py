"""Checkpoint save/load: weights + optimizer + scheduler + epoch + best + RNG."""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, opt, sched, epoch: int, best: float, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict() if opt is not None else None,
            "sched": sched.state_dict() if sched is not None else None,
            "epoch": epoch,
            "best": best,
            "rng": torch.get_rng_state(),
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path, model, opt=None, sched=None, map_location="cpu") -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None and ck.get("opt") is not None:
        opt.load_state_dict(ck["opt"])
    if sched is not None and ck.get("sched") is not None:
        sched.load_state_dict(ck["sched"])
    return ck
