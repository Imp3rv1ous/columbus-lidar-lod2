"""Phase 2: build sparse training labels on the subsampled cloud.

Project per-class annotation points onto the cloud with a guarded nearest-neighbor
match. Ground comes from USGS class-2 points; building, vegetation and other
annotations override it when their nearest match is within the distance guard.

Internal label codes:  0=unlabeled(ignore) 1=others 2=ground 3=vegetation 4=building
"""
from __future__ import annotations

import numpy as np
import laspy
from scipy.spatial import cKDTree

from ..config import Config, FT2M

# user label-file stem -> internal code
USER_CLASSES = {"building": 4, "vegetation": 3, "others": 1}


def assign_labels(
    sub_xyz: np.ndarray,
    is_ground: np.ndarray,
    annotations: list[tuple[str, int, np.ndarray]],
    guard: float,
) -> tuple[np.ndarray, dict]:
    """Pure label-assignment core (testable, no I/O).

    sub_xyz      (M,3) subsampled cloud
    is_ground    (M,)  bool, voxels that are ground from USGS class-2
    annotations  list of (name, code, ann_xyz(K,3)) user annotations in the SAME frame
    guard        max distance (m) for an annotation to claim its nearest voxel

    Ground is filled first; user annotations override it, nearest-annotation wins.
    """
    n_sub = sub_xyz.shape[0]
    labels = np.zeros(n_sub, dtype=np.int64)
    labels[is_ground] = 2

    tree = cKDTree(sub_xyz)
    sub_ids, codes, dists = [], [], []
    stats: dict = {}
    for name, code, ann_xyz in annotations:
        if ann_xyz.shape[0] == 0:
            stats[name] = {"kept": 0, "rejected": 0}
            continue
        dist, idx = tree.query(ann_xyz, k=1)
        keep = dist <= guard
        stats[name] = {"kept": int(keep.sum()), "rejected": int((~keep).sum())}
        sub_ids.append(idx[keep])
        codes.append(np.full(int(keep.sum()), code, dtype=np.int64))
        dists.append(dist[keep])

    if sub_ids:
        sub_ids = np.concatenate(sub_ids)
        codes = np.concatenate(codes)
        dists = np.concatenate(dists)
        order = np.argsort(dists, kind="stable")          # closest first
        uniq_sub, first = np.unique(sub_ids[order], return_index=True)
        labels[uniq_sub] = codes[order][first]            # nearest annotation claims the voxel

    return labels, stats


def _read_label_xyz_m(path, origin: np.ndarray) -> np.ndarray:
    las = laspy.read(str(path))
    xyz_m = np.column_stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
    ).astype(np.float64) * FT2M
    return xyz_m - origin  # into the tile-centered metric frame of the cache


def build_labels(cfg: Config, tile: str, guard: float | None = None):
    cache = np.load(cfg.cache_dir / f"{tile}.npz")
    sub_xyz = cache["sub_xyz"].astype(np.float64)
    inv = cache["inv"].astype(np.int64).reshape(-1)
    full_class = cache["full_class"]
    origin = cache["origin"]
    n_sub = sub_xyz.shape[0]
    if guard is None:
        guard = cfg.sub_grid * np.sqrt(3) * 1.5  # ~0.83 m; coords are aligned -> ~nothing rejected

    # ground = voxels whose majority of full points are ASPRS class 2
    cnt = np.bincount(inv, minlength=n_sub).astype(np.float64)
    gcnt = np.bincount(inv, weights=(full_class == 2).astype(np.float64), minlength=n_sub)
    is_ground = (gcnt / cnt) > 0.5

    annotations = []
    for name, code in USER_CLASSES.items():
        p = cfg.train_dir / tile / f"{tile}_{name}.las"
        axyz = _read_label_xyz_m(p, origin) if p.exists() else np.empty((0, 3))
        annotations.append((name, code, axyz))

    labels, stats = assign_labels(sub_xyz, is_ground, annotations, float(guard))
    return labels, stats, float(guard)
