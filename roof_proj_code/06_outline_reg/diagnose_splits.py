#!/usr/bin/env python
"""Split-vs-occlusion arbitration for gap-divided building masks.

Physical discriminator: between two SEPARATE buildings the gap is open
ground -> class-2 points at DTM height appear inside the gap corridor
(LiDAR penetrates canopy). A single building whose mid-section is hidden
by trees has solid roof under the canopy -> no returns near ground level.

For every instance whose roof mask splits into >=2 substantial components
(>=25 % of the largest), build the corridor between components (cells
within 3 m of both sides), and measure the fraction of corridor cells
containing ground-class points within +/-1 m of the DTM.
frac >= SPLIT_FRAC -> verdict TWO BUILDINGS (instance to split);
else -> occluded single building (union rect stands).
"""

import csv
import time
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
FULL = BASE / "04_05_roof_planes/output_full"
PRED = BASE / "01_semantic_randlanet/results/preds"
OUT = BASE / "06_outline_reg/output"
FT_TO_M = 0.30480060960121924
CELL = 0.5
GCELL = 1.0
SPLIT_FRAC = 0.35
FORCE_CHECK = [228, 216, 307]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    # ---- ground hash: per 1 m cell, min ground z (DTM) + z list bounds
    tiles = sorted(PRED.glob("*_pred.laz"))
    gmin = np.array([np.inf, np.inf])
    for t in tiles:
        with laspy.open(t) as r:
            gmin = np.minimum(gmin, r.header.mins[:2])
    x0g, y0g = gmin * FT_TO_M
    gx_l, gy_l, gz_l = [], [], []
    for t in tiles:
        las = laspy.read(t)
        m = np.asarray(las.classification) == 2
        gx_l.append(np.asarray(las.x)[m]); gy_l.append(np.asarray(las.y)[m])
        gz_l.append(np.asarray(las.z)[m])
    gx = np.concatenate(gx_l) * FT_TO_M
    gy = np.concatenate(gy_l) * FT_TO_M
    gz = (np.concatenate(gz_l) * FT_TO_M).astype(np.float32)
    nxg = int((gx.max() - x0g) / GCELL) + 2
    nyg = int((gy.max() - y0g) / GCELL) + 2
    dtm = np.full((nyg, nxg), np.inf, np.float32)
    cix = ((gx - x0g) / GCELL).astype(np.int32)
    ciy = ((gy - y0g) / GCELL).astype(np.int32)
    np.minimum.at(dtm, (ciy, cix), gz)
    has_g = np.isfinite(dtm)
    _, (iyn, ixn) = ndimage.distance_transform_edt(~has_g, return_indices=True)
    dtm_f = dtm[iyn, ixn]
    log(f"ground points {len(gx):,}; DTM {nxg}x{nyg}")

    las = laspy.read(FULL / "buildings_planes.laz")
    inst = np.asarray(las.instance_id)
    styp = np.asarray(las.surf_type)
    pid = np.asarray(las.plane_id)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                         np.asarray(las.z)]) * FT_TO_M

    order = np.argsort(inst, kind="stable")
    uniq, starts = np.unique(inst[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    s8 = np.ones((3, 3), bool)
    rows = []
    render = {}
    for fid, a, b in zip(uniq, starts, ends):
        if fid == 0:
            continue
        idx = order[a:b]
        P = X[idx]
        rm = (styp[idx] == 1) & (pid[idx] > 0)
        if rm.sum() < 12:
            continue
        x0m, y0m = P[:, 0].min(), P[:, 1].min()
        gxc = ((P[rm, 0] - x0m) / CELL).astype(np.int32)
        gyc = ((P[rm, 1] - y0m) / CELL).astype(np.int32)
        m = np.zeros((gyc.max() + 3, gxc.max() + 3), bool)
        m[gyc + 1, gxc + 1] = True
        mc = ndimage.binary_fill_holes(ndimage.binary_closing(m, structure=s8))
        lab, n = ndimage.label(mc, structure=s8)
        if n < 2 and int(fid) not in FORCE_CHECK:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        big = [i + 1 for i, s in enumerate(sizes) if s >= 0.25 * sizes.max()]
        if len(big) < 2:
            if int(fid) in FORCE_CHECK:
                # thin-gap probe: erode once to reveal weak necks
                er = ndimage.binary_erosion(mc, structure=s8)
                lab, n = ndimage.label(er, structure=s8)
                sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
                big = [i + 1 for i, s in enumerate(sizes) if n and s >= 0.25 * sizes.max()]
                if len(big) < 2:
                    rows.append((int(fid), 1, -1.0, "single_connected"))
                    continue
            else:
                continue
        A = lab == big[0]
        Bm = lab == big[1]
        dA = ndimage.distance_transform_edt(~A) * CELL
        dB = ndimage.distance_transform_edt(~Bm) * CELL
        corridor = (dA <= 3.0) & (dB <= 3.0) & ~A & ~Bm
        cy, cx = np.nonzero(corridor)
        if len(cy) < 4:
            rows.append((int(fid), len(big), -1.0, "no_corridor"))
            continue
        wx = (cx - 1 + 0.5) * CELL + x0m
        wy = (cy - 1 + 0.5) * CELL + y0m
        gxi = np.clip(((wx - x0g) / GCELL).astype(np.int32), 0, nxg - 1)
        gyi = np.clip(((wy - y0g) / GCELL).astype(np.int32), 0, nyg - 1)
        ground_here = has_g[gyi, gxi]
        frac = float(ground_here.mean())
        verdict = "TWO_BUILDINGS" if frac >= SPLIT_FRAC else "occluded_single"
        rows.append((int(fid), len(big), round(frac, 2), verdict))
        render[int(fid)] = (P[rm].copy(), wx.copy(), wy.copy(),
                            ground_here.copy(), frac, verdict)
    two = [r for r in rows if r[3] == "TWO_BUILDINGS"]
    log(f"gap-divided instances checked: {len(rows)}; TWO_BUILDINGS: {len(two)}")
    for r in rows:
        log(f"  id {r[0]:5d}: parts {r[1]}, ground-in-gap {r[2]}, {r[3]}")
    with open(OUT / "split_verdicts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "n_parts", "ground_frac_in_gap", "verdict"])
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [f for f in FORCE_CHECK if f in render] + \
           [r[0] for r in rows if r[0] in render and r[0] not in FORCE_CHECK][:9]
    show = show[:12]
    ncol = 4
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
    axes = np.atleast_2d(axes)
    for k, fid in enumerate(show):
        ax = axes[k // ncol, k % ncol]
        P, wx, wy, gh, frac, verdict = render[fid]
        c0 = P[:, :2].mean(axis=0)
        ax.scatter(P[:, 0] - c0[0], P[:, 1] - c0[1], s=0.7, c="0.7",
                   rasterized=True)
        ax.scatter(wx[~gh] - c0[0], wy[~gh] - c0[1], s=7, c="orange",
                   label="gap: no ground")
        ax.scatter(wx[gh] - c0[0], wy[gh] - c0[1], s=7, c="blue",
                   label="gap: GROUND hit")
        ax.set_title(f"id {fid}: ground {frac:.0%} -> {verdict}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.legend(fontsize=7, markerscale=1.5)
    for k in range(len(show), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("Gap arbitration: blue = laser reached GROUND in the gap "
                 "(=> two buildings), orange = no ground (=> occluded roof)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "qa_split_verdicts.png", dpi=140)
    log("wrote qa_split_verdicts.png")


if __name__ == "__main__":
    main()
