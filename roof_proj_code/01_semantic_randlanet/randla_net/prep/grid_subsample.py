"""Deterministic voxel-grid subsampling (barycenter), pure numpy.

Replaces the reference's compiled C++ `grid_subsampling` op. Same semantics:
one representative point per occupied `grid`-sized voxel = mean of the points
(and mean of their features) in that voxel.

Returns the subsampled cloud plus `inv` (full -> subsampled voxel index), which
doubles as the reprojection index used at inference time.
"""
from __future__ import annotations

import numpy as np


def grid_subsample(
    xyz: np.ndarray, feats: np.ndarray, grid: float = 0.32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # voxel coords (xyz is tile-centered >= 0, so indices are non-negative)
    vox = np.floor(xyz / grid).astype(np.int64)
    # hash 3 voxel indices into one int64 key (indices well under 2**20 for a 381 m tile)
    B = np.int64(1 << 20)
    key = (vox[:, 0] * B + vox[:, 1]) * B + vox[:, 2]

    _, inv = np.unique(key, return_inverse=True)
    inv = inv.astype(np.int64)
    n_vox = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n_vox).astype(np.float64)

    sub_xyz = np.empty((n_vox, xyz.shape[1]), dtype=np.float64)
    for d in range(xyz.shape[1]):
        sub_xyz[:, d] = np.bincount(inv, weights=xyz[:, d], minlength=n_vox) / cnt

    sub_feats = np.empty((n_vox, feats.shape[1]), dtype=np.float64)
    for d in range(feats.shape[1]):
        sub_feats[:, d] = np.bincount(inv, weights=feats[:, d], minlength=n_vox) / cnt

    return sub_xyz, sub_feats, inv
