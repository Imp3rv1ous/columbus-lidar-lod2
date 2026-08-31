"""list[Sample] -> Batch (stack along the batch dim, numpy -> torch)."""
from __future__ import annotations

import numpy as np
import torch

from ..common.types import Batch, MultiScaleIndices


def collate(samples) -> Batch:
    coords = torch.from_numpy(np.stack([s.coords for s in samples]))      # (B,N,3) f32
    feats = torch.from_numpy(np.stack([s.feats for s in samples]))        # (B,N,C) f32
    labels = torch.from_numpy(np.stack([s.labels for s in samples]))      # (B,N)   i64
    point_idx = torch.from_numpy(np.stack([s.point_idx for s in samples]))

    L = len(samples[0].ms.neighbors)

    def stack(attr, i):
        return torch.from_numpy(np.stack([getattr(s.ms, attr)[i] for s in samples]))

    ms = MultiScaleIndices(
        neighbors=[stack("neighbors", i) for i in range(L)],
        sub_idx=[stack("sub_idx", i) for i in range(L)],
        interp_idx=[stack("interp_idx", i) for i in range(L)],
    )
    return Batch(coords, feats, labels, ms, point_idx, [s.tile for s in samples])
