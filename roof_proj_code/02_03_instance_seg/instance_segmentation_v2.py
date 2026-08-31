#!/usr/bin/env python
"""Building instance segmentation v2 — fixes v1 under-segmentation.

v1 failure (verified by user in CloudCompare): 1 m cells + 3x3 closing +
8-connectivity bridge gaps up to ~2 m; side-yard spacing between detached
houses in this neighborhood is 1.5-2.5 m, so chains of separate houses merged
into one instance.

v2 changes:
  1. CELL 1.0 -> 0.5 m; NO morphological closing (the bridging culprit).
     Interior occlusion holes are filled per-component with binary_fill_holes,
     which cannot bridge across exterior gaps.
  2. Occupancy requires >= MIN_PTS_CELL points per cell -> stray misclassified
     tree points cannot form one-point bridges.
  3. Components that are still connected are tested for multiple "cores"
     (erosion by ~1.5 m). If >= 2 substantial cores exist, the component is
     split by marker-controlled watershed on the inverted max-height surface
     (boundary falls in the height valley between roofs) — the standard
     technique for separating weakly-connected objects; SI-DVDC (Li et al.
     2026) solves the same problem with density-reachability graphs.
  4. Anti-over-segmentation stages from v1 kept: fragment absorption
     (<20 m2 -> nearest >=50 m2 anchor within 3 m, |dz50| <= 3 m), noise
     filter, audit logs. New splits_log.csv records every watershed split.
  5. QA zoom rendered at the user's CloudCompare check point
     (Xg 1841781.79, Yg 711006.65 ftUS) where v1 produced instance_id 50.
"""

import sys
import time
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

# ---------------------------------------------------------------- parameters
PRED_DIR = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/01_semantic_randlanet/results/preds")
OUT_DIR = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2")

FT_TO_M = 0.30480060960121924

CELL = 0.5              # m
MIN_PTS_CELL = 2        # cells with fewer building points are not occupied
BUILDING_CLASS = 6

ERODE_R = 1.5           # m, core erosion radius (cuts contact bridges < ~3 m wide)
MIN_CORE_AREA = 6.0     # m2, a core must be at least this big to seed a split

NOISE_AREA = 20.0       # m2
BIG_MIN_AREA = 50.0     # m2
ABSORB_MAX_AREA = 20.0  # m2
ABSORB_DIST = 3.0       # m
ABSORB_MAX_DZ = 3.0     # m
ORPHAN_PT_DIST = 1.5    # m, points in dropped cells attach to labels within this

CHECK_FT = (1841781.790039, 711006.650024)  # user's CloudCompare pick (ftUS)
ZOOM_HALF = 200.0       # m, residential zoom half-size
CHECK_HALF = 120.0      # m, check-point zoom half-size


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_slices(sorted_ids):
    uniq, starts = np.unique(sorted_ids, return_index=True)
    ends = np.append(starts[1:], len(sorted_ids))
    return {u: slice(s, e) for u, s, e in zip(uniq, starts, ends)}


def disk(r_cells):
    y, x = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
    return (x * x + y * y) <= r_cells * r_cells


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = sorted(PRED_DIR.glob("*_pred.laz"))
    log(f"{len(tiles)} tiles")

    # ---------------------------------------------------------- load
    xs, ys, zs, tid = [], [], [], []
    src_header = None
    for i, t in enumerate(tiles):
        las = laspy.read(t)
        if src_header is None:
            src_header = las.header
        m = np.asarray(las.classification) == BUILDING_CLASS
        xs.append(np.asarray(las.x)[m])
        ys.append(np.asarray(las.y)[m])
        zs.append(np.asarray(las.z)[m])
        tid.append(np.full(int(m.sum()), i, dtype=np.uint8))
    x_ft = np.concatenate(xs); y_ft = np.concatenate(ys); z_ft = np.concatenate(zs)
    src_tile = np.concatenate(tid)
    del xs, ys, zs, tid
    n = len(x_ft)
    log(f"building points: {n:,}")

    xm = x_ft * FT_TO_M; ym = y_ft * FT_TO_M; zm = z_ft * FT_TO_M

    # ---------------------------------------------------------- rasterize
    x0, y0 = xm.min(), ym.min()
    ix = ((xm - x0) / CELL).astype(np.int32)
    iy = ((ym - y0) / CELL).astype(np.int32)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    log(f"grid {nx} x {ny} cells @ {CELL} m")

    cnt = np.zeros((ny, nx), np.uint16)
    np.add.at(cnt, (iy, ix), 1)
    occ = cnt >= MIN_PTS_CELL
    zg = np.full((ny, nx), -1e9, np.float32)
    np.maximum.at(zg, (iy, ix), zm.astype(np.float32))
    log(f"occupied cells: {occ.sum():,} (dropped speckle cells: "
        f"{int(((cnt > 0) & ~occ).sum()):,})")

    # ---------------------------------------------------------- CC + split
    s8 = np.ones((3, 3), bool)
    comps0, nc0 = ndimage.label(occ, structure=s8)
    log(f"raw components (no closing): {nc0}")
    objs = ndimage.find_objects(comps0)
    selem = disk(int(round(ERODE_R / CELL)))
    min_core_cells = int(MIN_CORE_AREA / (CELL * CELL))

    labels2 = np.zeros((ny, nx), np.int32)
    next_id = 1
    splits = []
    for ci in range(1, nc0 + 1):
        sl = objs[ci - 1]
        m = comps0[sl] == ci
        mf = ndimage.binary_fill_holes(m)
        core = ndimage.binary_erosion(mf, structure=selem) & m
        lab_c, k = ndimage.label(core, structure=s8)
        if k >= 2:
            sizes = np.bincount(lab_c.ravel())
            valid = [j for j in range(1, k + 1) if sizes[j] >= min_core_cells]
        else:
            valid = []
        if len(valid) >= 2:
            markers = np.zeros(m.shape, np.int32)
            for r, j in enumerate(valid, 1):
                markers[lab_c == j] = r
            ws = watershed(-zg[sl], markers=markers, mask=mf, connectivity=2)
            tgt = labels2[sl]
            good = m & (ws > 0)
            tgt[good] = ws[good] + next_id - 1
            leftover = m & (ws == 0)
            if leftover.any():          # unreachable pockets: own id
                tgt[leftover] = next_id + len(valid)
                extra = 1
            else:
                extra = 0
            part_areas = [float((good & (ws == r)).sum() * CELL * CELL)
                          for r in range(1, len(valid) + 1)]
            splits.append((ci, len(valid), part_areas))
            next_id += len(valid) + extra
        else:
            labels2[sl][m] = next_id
            next_id += 1
    nlab = next_id - 1
    log(f"after watershed splitting: {nlab} labels "
        f"({len(splits)} components split, "
        f"{sum(s[1] for s in splits) - len(splits)} extra instances)")

    # ---------------------------------------------------------- absorption
    cys, cxs = np.nonzero(labels2 > 0)
    clab = labels2[cys, cxs]
    order = np.argsort(clab, kind="stable")
    cys, cxs, clab_s = cys[order], cxs[order], clab[order]
    cell_sl = group_slices(clab_s)
    area = np.zeros(nlab + 1)
    for c, slc in cell_sl.items():
        area[c] = (slc.stop - slc.start) * CELL * CELL

    lab_p = labels2[iy, ix]                       # 0 for dropped cells
    porder = np.argsort(lab_p, kind="stable")
    p_sl = group_slices(lab_p[porder])
    z50 = np.zeros(nlab + 1)
    for c, slc in p_sl.items():
        if c > 0:
            z50[c] = np.median(zm[porder[slc]])

    big_ids = np.nonzero(area >= BIG_MIN_AREA)[0]
    small_ids = [c for c in cell_sl if 0 < area[c] < ABSORB_MAX_AREA]
    big_y = np.concatenate([cys[cell_sl[c]] for c in big_ids])
    big_x = np.concatenate([cxs[cell_sl[c]] for c in big_ids])
    big_l = np.concatenate([np.full(cell_sl[c].stop - cell_sl[c].start, c)
                            for c in big_ids])
    tree = cKDTree(np.column_stack([big_x * CELL, big_y * CELL]))
    remap = np.arange(nlab + 1)
    absorbed = 0
    for c in small_ids:
        slc = cell_sl[c]
        pts = np.column_stack([cxs[slc] * CELL, cys[slc] * CELL])
        d, idx = tree.query(pts, k=1)
        j = int(np.argmin(d))
        if d[j] <= ABSORB_DIST + CELL:
            targ = int(big_l[idx[j]])
            if abs(z50[c] - z50[targ]) <= ABSORB_MAX_DZ:
                remap[c] = targ
                absorbed += 1
    log(f"fragments <{ABSORB_MAX_AREA:.0f} m2: {len(small_ids)}, absorbed {absorbed}")

    lab_p = remap[lab_p]
    labels2 = remap[labels2]
    area2 = np.bincount(labels2[labels2 > 0].ravel(),
                        minlength=nlab + 1).astype(float) * CELL * CELL

    # ---------------------------------------------------------- final ids
    keep = np.nonzero(area2 >= NOISE_AREA)[0]
    keep = keep[keep > 0]
    keep = keep[np.argsort(-area2[keep])]
    final_id = np.zeros(nlab + 1, np.uint32)
    final_id[keep] = np.arange(1, len(keep) + 1, dtype=np.uint32)
    inst = final_id[lab_p]
    final_grid = final_id[labels2]

    # orphan points (dropped cells): nearest labeled cell within ORPHAN_PT_DIST
    orphan = inst == 0
    if orphan.any():
        dist, (ny_i, nx_i) = ndimage.distance_transform_edt(
            final_grid == 0, return_indices=True, sampling=CELL)
        oy, ox = iy[orphan], ix[orphan]
        od = dist[oy, ox]
        olab = final_grid[ny_i[oy, ox], nx_i[oy, ox]]
        olab[od > ORPHAN_PT_DIST] = 0
        inst[orphan] = olab
    ninst = len(keep)
    log(f"final instances: {ninst} (noise points {(inst == 0).sum():,} = "
        f"{(inst == 0).mean()*100:.2f}%)")

    # ---------------------------------------------------------- stats csv
    import csv
    porder = np.argsort(inst, kind="stable")
    p_sl2 = group_slices(inst[porder])
    rows = []
    for fid, slc in p_sl2.items():
        if fid == 0:
            continue
        idxp = porder[slc]
        zi = zm[idxp]; xi = xm[idxp]; yi = ym[idxp]
        npt = len(idxp)
        a = float(area2[keep[fid - 1]])
        xy = np.column_stack([xi - xi.mean(), yi - yi.mean()])
        cov = (xy.T @ xy) / max(npt - 1, 1)
        ev = np.linalg.eigvalsh(cov)
        ar = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
        rows.append((fid, npt, a, npt / a if a > 0 else 0,
                     float(xi.mean()), float(yi.mean()),
                     float(np.percentile(zi, 5)), float(np.median(zi)),
                     float(np.percentile(zi, 95)), ar,
                     len(np.unique(src_tile[idxp]))))
    rows.sort(key=lambda r: r[0])
    with open(OUT_DIR / "instances.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "n_points", "area_m2", "density_pt_m2",
                    "cx_m", "cy_m", "z05_m", "z50_m", "z95_m",
                    "aspect_ratio", "n_tiles"])
        w.writerows(rows)
    with open(OUT_DIR / "splits_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "n_parts", "part_areas_m2"])
        for ci, k, pa in splits:
            w.writerow([ci, k, ";".join(f"{a:.0f}" for a in pa)])

    areas = np.array([r[2] for r in rows]); ars = np.array([r[9] for r in rows])
    resi = int(np.sum((areas >= 50) & (areas <= 400) & (ars < 3)))
    log(f"area median {np.median(areas):.0f} m2; residential-geometry "
        f"(50-400 m2, AR<3): {resi}; multi-tile: {sum(1 for r in rows if r[10] > 1)}")

    # ---------------------------------------------------------- check point
    cxq = CHECK_FT[0] * FT_TO_M - x0
    cyq = CHECK_FT[1] * FT_TO_M - y0
    qx1, qx2 = int((cxq - 40) / CELL), int((cxq + 40) / CELL)
    qy1, qy2 = int((cyq - 40) / CELL), int((cyq + 40) / CELL)
    win = final_grid[max(qy1, 0):qy2, max(qx1, 0):qx2]
    ids_here = np.unique(win[win > 0])
    log(f"check point ±40 m: {len(ids_here)} instances: {ids_here.tolist()}")

    # ---------------------------------------------------------- LAZ
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = src_header.scales
    header.offsets = src_header.offsets
    for vlr in src_header.vlrs:
        header.vlrs.append(vlr)
    header.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.uint32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="src_tile", type=np.uint8))
    out = laspy.LasData(header)
    out.x = x_ft; out.y = y_ft; out.z = z_ft
    out.classification = np.full(n, BUILDING_CLASS, np.uint8)
    out.instance_id = inst
    out.src_tile = src_tile
    out.write(OUT_DIR / "buildings_instances.laz")
    log("wrote LAZ")

    # ---------------------------------------------------------- QA images
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    lut = rng.integers(0, 2**24, size=ninst + 1, dtype=np.int64)
    lut[0] = 0xFFFFFF
    v = lut[final_grid]
    rgb = np.zeros((*final_grid.shape, 3), np.uint8)
    rgb[..., 0] = (v >> 16) & 255; rgb[..., 1] = (v >> 8) & 255; rgb[..., 2] = v & 255

    def save_crop(cx, cy, half, name, title):
        x1, x2 = int(max(cx - half, 0) / CELL), int(min(cx + half, nx * CELL) / CELL)
        y1, y2 = int(max(cy - half, 0) / CELL), int(min(cy + half, ny * CELL) / CELL)
        fig, ax = plt.subplots(figsize=(11, 11))
        ax.imshow(rgb[y1:y2, x1:x2], origin="lower", interpolation="nearest",
                  extent=[x1 * CELL, x2 * CELL, y1 * CELL, y2 * CELL])
        ax.set_title(title)
        ax.set_xlabel("x (m, local)"); ax.set_ylabel("y (m, local)")
        fig.tight_layout(); fig.savefig(OUT_DIR / name, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 11))
    step = max(1, int(1.0 / CELL))
    ax.imshow(rgb[::step, ::step], origin="lower", interpolation="nearest")
    ax.set_title(f"v2 building instances — full extent ({ninst} instances, "
                 f"cell {CELL} m, no closing, watershed split)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "qa_overview.png", dpi=160)
    plt.close(fig)

    save_crop(cxq, cyq, CHECK_HALF, "qa_checkpoint.png",
              "v2 @ CloudCompare check point (v1 merged this into one instance)")
    # same residential window as v1 zoom for comparison
    save_crop(cxq, cyq, ZOOM_HALF, "qa_zoom.png",
              "v2 building instances — 400 x 400 m around check point")
    log("wrote QA images; done")


if __name__ == "__main__":
    main()
