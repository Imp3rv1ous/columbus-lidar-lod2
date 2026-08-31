"""Possibility-based spatially-regular block sampler (RandLA-Net pattern).

Keeps a per-point "possibility" so that block centers spread to cover every tile
uniformly over an epoch. Stateful; each DataLoader worker holds its own copy and
explores independently.
"""
from __future__ import annotations

import numpy as np


class SpatiallyRegularSampler:
    def __init__(self, tiles: list[dict], num_points: int, noise_init: float, seed: int = 0):
        # each tile dict: {"xyz": (M,3), "tree": cKDTree}
        self.tiles = tiles
        self.num_points = num_points
        self.noise = noise_init / 10.0
        self.rng = np.random.RandomState(seed)
        self.possibility = [self.rng.rand(t["xyz"].shape[0]) * 1e-3 for t in tiles]
        self.min_poss = [float(p.min()) for p in self.possibility]

    def next_block(self) -> tuple[int, np.ndarray]:
        """Return (tile_index, point_indices) for one block of num_points points."""
        ti = int(np.argmin(self.min_poss))
        xyz = self.tiles[ti]["xyz"]
        tree = self.tiles[ti]["tree"]

        center_i = int(np.argmin(self.possibility[ti]))
        center = xyz[center_i] + self.rng.randn(3) * self.noise

        n = min(self.num_points, xyz.shape[0])
        idx = tree.query(center[None, :], k=n)[1][0]

        # bump possibility near the center so it is less likely re-picked
        d = np.sum((xyz[idx] - center) ** 2, axis=1)
        delta = np.square(1.0 - d / (d.max() + 1e-9))
        self.possibility[ti][idx] += delta
        self.min_poss[ti] = float(self.possibility[ti].min())

        self.rng.shuffle(idx)  # shuffle so "first N//ratio" is a random subsample downstream
        return ti, idx
