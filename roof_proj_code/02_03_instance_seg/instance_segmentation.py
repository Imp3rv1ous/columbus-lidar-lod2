#!/usr/bin/env python
"""Building instance segmentation from RandLA-Net classified point clouds.

Route: footprint-free clustering (detached-residential regime).
  1. Read all 19 pred tiles, keep class 6 (building) points, merge in memory
     (eliminates cross-tile building splits by construction).
  2. Rasterize to a 1 m occupancy grid (working units: meters; source is ftUS).
  3. Morphological closing (3x3) to bridge small occlusion gaps.
  4. 8-connected component labeling  -> candidate instances.
  5. Asymmetric fragment absorption (anti-over-segmentation):
     components < ABSORB_MAX_AREA m2 that lie within ABSORB_DIST m of a
     component >= BIG_MIN_AREA m2 and match its roof height within
     ABSORB_MAX_DZ m are merged INTO the big one. Big components are never
     merged with each other, so two real adjacent houses cannot fuse here.
     20-50 m2 components are candidate accessory buildings (garage/shed):
     kept separate, flagged `near_big` for QA.
  6. Components < NOISE_AREA m2 after absorption -> instance_id 0 (noise).
  7. Outputs (all coordinates kept in the original ftUS CRS):
     - buildings_instances.laz  (class-6 points + instance_id + src_tile)
     - instances.csv            (per-instance attributes)
     - qa_overview.png / qa_zoom.png (colored by instance)
     - summary printed to stdout
"""

import sys
import time
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- parameters
PRED_DIR = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/01_semantic_randlanet/results/preds")
OUT_DIR = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output")

FT_TO_M = 0.30480060960121924  # US survey foot

CELL = 1.0            # m, raster cell size
CLOSE_SIZE = 3        # closing structuring element (cells)
BUILDING_CLASS = 6

NOISE_AREA = 20.0     # m2, below this (after absorption) -> noise (id 0)
BIG_MIN_AREA = 50.0   # m2, absorption anchors must be at least this big
ABSORB_MAX_AREA = 20.0  # m2, only fragments smaller than this get absorbed
ABSORB_DIST = 3.0     # m, max gap between fragment and anchor (edge-to-edge approx)
ABSORB_MAX_DZ = 3.0   # m, max |z50 fragment - z50 anchor|
FLAG_NEAR_DIST = 3.0  # m, 20-50 m2 comps within this of a big comp get flagged

ZOOM_HALF = 200.0     # m, half-size of the QA zoom window


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_slices(sorted_ids):
    """Return {id: slice} for a sorted id array."""
    uniq, starts = np.unique(sorted_ids, return_index=True)
    ends = np.append(starts[1:], len(sorted_ids))
    return {u: slice(s, e) for u, s, e in zip(uniq, starts, ends)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = sorted(PRED_DIR.glob("*_pred.laz"))
    if not tiles:
        sys.exit(f"no tiles found in {PRED_DIR}")
    log(f"{len(tiles)} tiles")

    # ---------------------------------------------------------- 1. load
    xs, ys, zs, tid = [], [], [], []
    src_header = None
    total_pts = 0
    for i, t in enumerate(tiles):
        las = laspy.read(t)
        if src_header is None:
            src_header = las.header
        cls = np.asarray(las.classification)
        m = cls == BUILDING_CLASS
        total_pts += len(cls)
        xs.append(np.asarray(las.x)[m])
        ys.append(np.asarray(las.y)[m])
        zs.append(np.asarray(las.z)[m])
        tid.append(np.full(int(m.sum()), i, dtype=np.uint8))
        log(f"  {t.name}: {len(cls):,} pts, building {int(m.sum()):,}")
    x_ft = np.concatenate(xs); y_ft = np.concatenate(ys); z_ft = np.concatenate(zs)
    src_tile = np.concatenate(tid)
    del xs, ys, zs, tid
    n = len(x_ft)
    log(f"total {total_pts:,} pts, building {n:,} ({n/total_pts*100:.1f}%)")

    # meters for processing; original ftUS kept for output
    xm = x_ft * FT_TO_M
    ym = y_ft * FT_TO_M
    zm = z_ft * FT_TO_M

    # ---------------------------------------------------------- 2. rasterize
    x0, y0 = xm.min(), ym.min()
    ix = ((xm - x0) / CELL).astype(np.int32)
    iy = ((ym - y0) / CELL).astype(np.int32)
    nx, ny = ix.max() + 1, iy.max() + 1
    log(f"grid {nx} x {ny} cells ({nx*CELL:.0f} x {ny*CELL:.0f} m)")
    occ = np.zeros((ny, nx), bool)
    occ[iy, ix] = True

    # ---------------------------------------------------------- 3-4. close + label
    closed = ndimage.binary_closing(occ, structure=np.ones((CLOSE_SIZE, CLOSE_SIZE), bool))
    closed |= occ  # closing must never delete occupied cells
    labels, ncomp = ndimage.label(closed, structure=np.ones((3, 3), bool))
    log(f"connected components: {ncomp}")

    comp_of_point = labels[iy, ix]  # every occupied cell survived -> always > 0

    # per-component cell lists (original occupied cells only -> true area)
    cys, cxs = np.nonzero(occ)
    clab = labels[cys, cxs]
    order = np.argsort(clab, kind="stable")
    cys, cxs, clab_s = cys[order], cxs[order], clab[order]
    cell_sl = group_slices(clab_s)

    area = np.zeros(ncomp + 1)
    for c, sl in cell_sl.items():
        area[c] = (sl.stop - sl.start) * CELL * CELL

    # per-component z50 from points
    porder = np.argsort(comp_of_point, kind="stable")
    p_sorted_comp = comp_of_point[porder]
    p_sl = group_slices(p_sorted_comp)
    z50 = np.zeros(ncomp + 1)
    for c, sl in p_sl.items():
        z50[c] = np.median(zm[porder[sl]])

    # ---------------------------------------------------------- 5. absorption
    big_ids = np.nonzero(area >= BIG_MIN_AREA)[0]
    small_ids = [c for c in cell_sl if 0 < area[c] < ABSORB_MAX_AREA]
    mid_ids = [c for c in cell_sl if ABSORB_MAX_AREA <= area[c] < BIG_MIN_AREA]

    big_cells_y = np.concatenate([cys[cell_sl[c]] for c in big_ids])
    big_cells_x = np.concatenate([cxs[cell_sl[c]] for c in big_ids])
    big_cells_lab = np.concatenate(
        [np.full(cell_sl[c].stop - cell_sl[c].start, c) for c in big_ids])
    tree = cKDTree(np.column_stack([big_cells_x * CELL, big_cells_y * CELL]))

    remap = np.arange(ncomp + 1)
    absorbed, rejected_dz = [], []
    for c in small_ids:
        sl = cell_sl[c]
        pts = np.column_stack([cxs[sl] * CELL, cys[sl] * CELL])
        d, idx = tree.query(pts, k=1)
        j = int(np.argmin(d))
        if d[j] <= ABSORB_DIST + CELL:  # cell-center distance ~ edge distance + <=1 cell
            target = int(big_cells_lab[idx[j]])
            if abs(z50[c] - z50[target]) <= ABSORB_MAX_DZ:
                remap[c] = target
                absorbed.append((c, target, float(d[j]), float(area[c])))
            else:
                rejected_dz.append(c)
    log(f"fragments <{ABSORB_MAX_AREA:.0f} m2: {len(small_ids)}; absorbed {len(absorbed)}; "
        f"near-but-height-rejected {len(rejected_dz)}")

    flagged_near = []
    for c in mid_ids:
        sl = cell_sl[c]
        pts = np.column_stack([cxs[sl] * CELL, cys[sl] * CELL])
        d, _ = tree.query(pts, k=1)
        if d.min() <= FLAG_NEAR_DIST + CELL:
            flagged_near.append(c)
    log(f"accessory candidates 20-50 m2: {len(mid_ids)}, of which near a big comp: {len(flagged_near)}")

    comp_of_point = remap[comp_of_point]
    clab_final = remap[clab]

    # recompute area on final labels
    area2 = np.bincount(clab_final, minlength=ncomp + 1).astype(float) * CELL * CELL

    # ---------------------------------------------------------- 6. final ids
    keep = np.nonzero(area2 >= NOISE_AREA)[0]
    keep = keep[keep > 0]
    keep = keep[np.argsort(-area2[keep])]  # by area desc
    final_id = np.zeros(ncomp + 1, dtype=np.uint32)
    final_id[keep] = np.arange(1, len(keep) + 1, dtype=np.uint32)
    inst = final_id[comp_of_point]
    log(f"final instances: {len(keep)} (noise points: {(inst == 0).sum():,} = "
        f"{(inst == 0).mean()*100:.2f}%)")

    # ---------------------------------------------------------- 7a. stats csv
    porder = np.argsort(inst, kind="stable")
    p_sl2 = group_slices(inst[porder])
    flag_near_final = {int(final_id[c]) for c in flagged_near if final_id[c] > 0}

    rows = []
    for fid, sl in p_sl2.items():
        if fid == 0:
            continue
        idxp = porder[sl]
        zi = zm[idxp]
        xi, yi = xm[idxp], ym[idxp]
        npt = len(idxp)
        a = float(area2[keep[fid - 1]])
        # oriented aspect ratio from point PCA (2D)
        xy = np.column_stack([xi - xi.mean(), yi - yi.mean()])
        cov = (xy.T @ xy) / max(npt - 1, 1)
        ev = np.linalg.eigvalsh(cov)
        ar = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
        ntiles = len(np.unique(src_tile[idxp]))
        rows.append((fid, npt, a, npt / a if a > 0 else 0.0,
                     float(xi.mean()), float(yi.mean()),
                     float(np.percentile(zi, 5)), float(np.median(zi)),
                     float(np.percentile(zi, 95)), ar, ntiles,
                     int(fid in flag_near_final)))
    rows.sort(key=lambda r: r[0])
    import csv
    with open(OUT_DIR / "instances.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "n_points", "area_m2", "density_pt_m2",
                    "cx_m", "cy_m", "z05_m", "z50_m", "z95_m",
                    "aspect_ratio", "n_tiles", "accessory_near_big"])
        w.writerows(rows)

    areas = np.array([r[2] for r in rows])
    ars = np.array([r[9] for r in rows])
    multi_tile = sum(1 for r in rows if r[10] > 1)
    resi = np.sum((areas >= 50) & (areas <= 400) & (ars < 3))
    log(f"area median {np.median(areas):.0f} m2, IQR {np.percentile(areas,25):.0f}-"
        f"{np.percentile(areas,75):.0f}; multi-tile instances {multi_tile}; "
        f"residential-geometry (50-400 m2, AR<3): {resi}")

    # ---------------------------------------------------------- 7b. LAZ output
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = src_header.scales
    header.offsets = src_header.offsets
    for vlr in src_header.vlrs:  # carry CRS WKT etc.
        header.vlrs.append(vlr)
    header.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.uint32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="src_tile", type=np.uint8))
    out = laspy.LasData(header)
    out.x = x_ft; out.y = y_ft; out.z = z_ft
    out.classification = np.full(n, BUILDING_CLASS, dtype=np.uint8)
    out.instance_id = inst
    out.src_tile = src_tile
    out.write(OUT_DIR / "buildings_instances.laz")
    log(f"wrote {OUT_DIR/'buildings_instances.laz'}")

    with open(OUT_DIR / "tile_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_tile", "file"])
        for i, t in enumerate(tiles):
            w.writerow([i, t.name])

    # ---------------------------------------------------------- 7c. QA images
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_grid = final_id[remap[labels]]
    rng = np.random.default_rng(42)
    lut = rng.integers(0, 2**24, size=len(keep) + 1, dtype=np.int64)
    lut[0] = 0xFFFFFF  # background/noise -> white
    rgb = np.zeros((*final_grid.shape, 3), dtype=np.uint8)
    v = lut[final_grid]
    rgb[..., 0] = (v >> 16) & 255
    rgb[..., 1] = (v >> 8) & 255
    rgb[..., 2] = v & 255

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_title(f"Building instances — full extent ({len(keep)} instances, "
                 f"cell {CELL} m)")
    ax.set_xlabel("x (m, local)"); ax.set_ylabel("y (m, local)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qa_overview.png", dpi=160)
    plt.close(fig)

    # zoom: densest residential window = window with most instances of 50-400 m2
    cxs_r = np.array([r[4] for r in rows]) - x0
    cys_r = np.array([r[5] for r in rows]) - y0
    resi_m = (areas >= 50) & (areas <= 400) & (ars < 3)
    if resi_m.sum() > 10:
        hx = np.clip(cxs_r[resi_m], 0, nx * CELL - 1)
        hy = np.clip(cys_r[resi_m], 0, ny * CELL - 1)
        H, xe, ye = np.histogram2d(hx, hy, bins=[max(nx // 400, 1), max(ny // 400, 1)])
        bi = np.unravel_index(np.argmax(H), H.shape)
        cx = (xe[bi[0]] + xe[bi[0] + 1]) / 2
        cy = (ye[bi[1]] + ye[bi[1] + 1]) / 2
    else:
        cx, cy = nx * CELL / 2, ny * CELL / 2
    x1, x2 = int(max(cx - ZOOM_HALF, 0) / CELL), int(min(cx + ZOOM_HALF, nx * CELL) / CELL)
    y1, y2 = int(max(cy - ZOOM_HALF, 0) / CELL), int(min(cy + ZOOM_HALF, ny * CELL) / CELL)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(rgb[y1:y2, x1:x2], origin="lower", interpolation="nearest",
              extent=[x1 * CELL, x2 * CELL, y1 * CELL, y2 * CELL])
    ax.set_title("Building instances — residential zoom (400 x 400 m)")
    ax.set_xlabel("x (m, local)"); ax.set_ylabel("y (m, local)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qa_zoom.png", dpi=160)
    plt.close(fig)
    log("wrote QA images")

    # absorption audit
    with open(OUT_DIR / "absorption_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fragment_comp", "target_comp", "gap_m", "fragment_area_m2"])
        w.writerows(absorbed)
    log("done")


if __name__ == "__main__":
    main()
