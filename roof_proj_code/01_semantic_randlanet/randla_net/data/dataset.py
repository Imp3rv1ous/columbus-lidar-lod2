"""ColumbusDataset: cached sub-clouds -> training blocks (Sample).

Map-style Dataset whose length is the number of blocks per epoch; each
__getitem__ draws one possibility-sampled block, augments it, and builds the
per-layer neighbourhood indices.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from ..config import Config
from ..common.types import Sample
from .sampler import SpatiallyRegularSampler
from .transforms import build_transforms
from .multiscale import build_multiscale


class ColumbusDataset(Dataset):
    def __init__(self, cfg: Config, tiles, split: str = "train", steps: int = 500, seed: int = 0):
        self.cfg = cfg
        self.split = split
        self.steps = steps
        self._seed = seed
        self.rng = np.random.RandomState(seed)

        self.tiles = []
        tile_info = []
        for t in tiles:
            d = np.load(cfg.cache_dir / f"{t}.npz")
            xyz = d["sub_xyz"].astype(np.float64)
            feats = d["sub_feats"].astype(np.float32)
            labels = np.load(cfg.cache_dir / f"{t}_labels.npy").astype(np.int64)
            tree = cKDTree(xyz)
            self.tiles.append({"name": t, "xyz": xyz, "feats": feats, "labels": labels, "tree": tree})
            tile_info.append({"xyz": xyz, "tree": tree})

        self.sampler = SpatiallyRegularSampler(tile_info, cfg.num_points, cfg.noise_init, seed)
        self.transform = build_transforms(augment=(split == "train"), rng=self.rng)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, _) -> Sample:
        ti, idx = self.sampler.next_block()
        T = self.tiles[ti]
        # Skip blocks with fewer than two valid classes so rare-class gradients
        # are not drowned by ground-only or single-class steps. The retry count
        # is capped to avoid an infinite loop.
        if self.split == "train":
            for _try in range(20):
                lab = T["labels"][idx]
                if np.unique(lab[lab != 0]).size >= 2:
                    break
                ti, idx = self.sampler.next_block()
                T = self.tiles[ti]
        coords = T["xyz"][idx].astype(np.float64)
        feats = T["feats"][idx].astype(np.float32)
        labels = T["labels"][idx].astype(np.int64)

        coords, feats = self.transform(coords, feats)
        ms = build_multiscale(coords, self.cfg.num_layers, self.cfg.k_n, self.cfg.sub_ratios)

        return Sample(
            coords=coords.astype(np.float32),
            feats=feats.astype(np.float32),
            labels=labels,
            ms=ms,
            point_idx=idx.astype(np.int64),
            tile=T["name"],
        )

    def reseed(self, base: int) -> None:
        """Re-seed this (forked) copy so each DataLoader worker explores a UNIQUE
        block trajectory instead of replaying the same seeded sequence — otherwise
        every worker emits the same blocks, shrinking effective coverage / eval-N."""
        self.rng = np.random.RandomState(base)
        self.transform = build_transforms(augment=(self.split == "train"), rng=self.rng)
        s = self.sampler
        s.rng = np.random.RandomState(base)
        s.possibility = [s.rng.rand(t["xyz"].shape[0]) * 1e-3 for t in s.tiles]
        s.min_poss = [float(p.min()) for p in s.possibility]
