"""Full-tile inference: spatially-regular blocks -> smoothed vote -> reproject.

Blocks are drawn with the same possibility sampler as training; per-sub-point
softmax probs are accumulated with exponential smoothing across overlapping
blocks until every point is seen >= min_votes times, then reprojected from the
subsampled cloud to full resolution via the cached `inv` index.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..common.types import Sample
from ..data.multiscale import build_multiscale
from ..data.transforms import RecenterBlock
from ..data.collate import collate


def infer_tile(cfg, model, cache_path, device, num_classes=4,
               min_votes=3, smooth=0.95, seed=0, max_blocks=100000, log_every=50):
    d = np.load(cache_path)
    sub_xyz = d["sub_xyz"].astype(np.float64)
    sub_feats = d["sub_feats"].astype(np.float32)
    inv = d["inv"].astype(np.int64)
    M = sub_xyz.shape[0]

    tree = cKDTree(sub_xyz)
    rng = np.random.RandomState(seed)
    probs = np.zeros((M, num_classes), np.float32)
    seen = np.zeros(M, np.int32)
    possibility = rng.rand(M) * 1e-3
    recenter = RecenterBlock()
    n_pts = min(cfg.num_points, M)

    model.eval()
    nb = 0
    while seen.min() < min_votes and nb < max_blocks:
        ci = int(np.argmin(possibility))
        center = sub_xyz[ci] + rng.randn(3) * (cfg.noise_init / 10.0)
        idx = tree.query(center[None, :], k=n_pts)[1][0]
        # CRITICAL train/serve fix: training shuffles block points (sampler.py) so
        # the multiscale "first N//ratio = random subsample" holds. KDTree returns
        # points distance-sorted; without this shuffle every coarse layer keeps only
        # the block center, dropping edge/clustered buildings -> smeared to veg.
        rng.shuffle(idx)

        coords, feats = recenter(sub_xyz[idx].astype(np.float64), sub_feats[idx])
        ms = build_multiscale(coords, cfg.num_layers, cfg.k_n, cfg.sub_ratios)
        s = Sample(coords.astype(np.float32), feats.astype(np.float32),
                   np.zeros(len(idx), np.int64), ms, idx.astype(np.int64), "infer")
        batch = collate([s]).to(device)
        with torch.no_grad():
            p = torch.softmax(model(batch)[0], dim=-1).cpu().numpy()  # (n_pts, C)

        probs[idx] = smooth * probs[idx] + (1.0 - smooth) * p
        seen[idx] += 1
        d2 = np.sum((sub_xyz[idx] - center) ** 2, axis=1)
        possibility[idx] += np.square(1.0 - d2 / (d2.max() + 1e-9))
        nb += 1
        if log_every and nb % log_every == 0:
            print(f"    block {nb}: min_seen={seen.min()} covered={100*np.mean(seen>0):.1f}%")

    n_unseen = int((seen == 0).sum())
    if n_unseen:
        print(f"    [warn] {n_unseen}/{M} sub-points never voted "
              f"(coverage {100*np.mean(seen > 0):.2f}%); they default to argmax(zeros)=others")

    sub_pred = probs.argmax(1) + 1          # 0..C-1 -> internal 1..C
    full_pred = sub_pred[inv]               # reproject subsampled -> full resolution
    return full_pred, seen, nb
