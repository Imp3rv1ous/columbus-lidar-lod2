"""Masked, class-weighted cross-entropy for partial labels.

Labels use internal codes 0..C where 0 is the ignored/unlabeled bin and 1..C are
the valid classes. Loss is computed only on labeled points; valid codes are
remapped to 0..C-1 to index the C-way logits.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def masked_weighted_ce(logits, labels, weight=None, ignore_index: int = 0):
    Cout = logits.shape[-1]
    lg = logits.reshape(-1, Cout)
    lb = labels.reshape(-1)
    valid = lb != ignore_index
    if valid.sum() == 0:
        return lg.sum() * 0.0
    tgt = lb[valid] - 1  # 1..C -> 0..C-1
    return F.cross_entropy(lg[valid], tgt, weight=weight)


def inverse_sqrt_weights(counts: dict, num_classes: int) -> torch.Tensor:
    """counts: {valid_code(1..C): n}. Returns (C,) weights, mean-normalized."""
    c = np.array([counts.get(i, 0) for i in range(1, num_classes + 1)], dtype=np.float64)
    freq = c / max(c.sum(), 1.0)
    w = 1.0 / np.sqrt(freq + 0.02)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
