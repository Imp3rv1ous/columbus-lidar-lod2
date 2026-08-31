"""Build the per-layer neighbourhood indices used by the RandLA-Net encoder/decoder.

Mirrors the reference `tf_map` exactly, but with scipy KDTree (CPU, off the GPU):
for each layer i, given the current points P_i (already shuffled so the first
N_{i+1} are a random subset):
  neighbors[i]  = KNN(P_i, P_i, k)                      (N_i, k)
  sub_idx[i]    = neighbors[i][:N_{i+1}]                (N_{i+1}, k)  pooling
  interp_idx[i] = nearest-of-P_{i+1} for each P_i       (N_i, 1)      upsample
  P_{i+1}       = P_i[:N_{i+1}]
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..common.types import MultiScaleIndices


def build_multiscale(xyz: np.ndarray, num_layers: int, k_n: int, sub_ratios) -> MultiScaleIndices:
    neighbors, sub_idx, interp_idx = [], [], []
    cur = xyz
    for i in range(num_layers):
        n = cur.shape[0]
        tree = cKDTree(cur)
        k = min(k_n, n)                                          # clamp: can't have >n neighbours
        nb = tree.query(cur, k=k)[1]
        if nb.ndim == 1:
            nb = nb[:, None]
        nb = nb.astype(np.int64)                                 # (n, k)
        n_sub = max(1, n // sub_ratios[i])
        sub_pts = cur[:n_sub]
        pool = nb[:n_sub]                                        # (n_sub, k)
        up = cKDTree(sub_pts).query(cur, k=1)[1].astype(np.int64)  # (n,)
        neighbors.append(nb)
        sub_idx.append(pool)
        interp_idx.append(up[:, None])
        cur = sub_pts
    return MultiScaleIndices(neighbors, sub_idx, interp_idx)
