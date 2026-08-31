"""Height-above-ground (HAG) from the USGS-classified ground points.

We build a coarse DTM by binning the class-2 ground points into `cell`-sized
cells (mean ground elevation per cell), then look up each point's local ground
height via inverse-distance-weighted k-NN over the occupied cell centers.
This is robust under canopy (cells aggregate sparse ground returns) and O(N).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_hag(
    xyz: np.ndarray,
    classification: np.ndarray,
    cell: float = 1.0,
    k: int = 4,
) -> np.ndarray:
    ground = classification == 2  # ASPRS ground
    if ground.sum() < k:
        # degenerate tile: fall back to height above global min
        return xyz[:, 2] - xyz[:, 2].min()

    g = xyz[ground]
    gi = np.floor(g[:, :2] / cell).astype(np.int64)
    cells, inv = np.unique(gi, axis=0, return_inverse=True)

    cz = np.bincount(inv, weights=g[:, 2])
    cnt = np.bincount(inv)
    cz = cz / cnt                      # mean ground z per occupied cell
    centers = (cells + 0.5) * cell     # (M,2) cell-center xy

    tree = cKDTree(centers)
    kk = min(k, len(centers))
    dist, idx = tree.query(xyz[:, :2], k=kk)
    if kk == 1:
        ground_z = cz[idx]
    else:
        w = 1.0 / (dist + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        ground_z = (cz[idx] * w).sum(axis=1)

    return xyz[:, 2] - ground_z
