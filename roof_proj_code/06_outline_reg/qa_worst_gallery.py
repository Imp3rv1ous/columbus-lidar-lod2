#!/usr/bin/env python
"""Render the worst regularized outlines over their roof points to
classify failure causes (wrong theta, tilted theta, rect prior misuse)."""

import csv
import math
import time
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage
from skimage import measure

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
FULL = BASE / "04_05_roof_planes/output_full"
OUT = BASE / "06_outline_reg/output"
FT_TO_M = 0.30480060960121924
CELL = 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_wkt(w):
    body = w.split("((")[1].split("))")[0]
    pts = []
    for tok in body.split(","):
        x, y = tok.strip().split()
        pts.append([float(x) * FT_TO_M, float(y) * FT_TO_M])
    return np.array(pts)


def main():
    rows = list(csv.DictReader(open(OUT / "outlines_regularized.csv")))
    by_id = {int(r["instance_id"]): r for r in rows}
    reg = [r for r in rows if r["method"] == "regular" and r["mean_offset_m"]]
    rect = [r for r in rows if r["method"] == "rect_prior" and r["mean_offset_m"]]
    reg.sort(key=lambda r: -float(r["mean_offset_m"]))
    rect.sort(key=lambda r: -float(r["mean_offset_m"]))
    lowo = sorted((r for r in rows if r["ortho_share"]),
                  key=lambda r: float(r["ortho_share"]))
    pick, seen = [], set()
    for group, k in ((rect, 6), (reg, 6), (lowo, 4)):
        c = 0
        for r in group:
            fid = int(r["instance_id"])
            if fid in seen:
                continue
            seen.add(fid); pick.append(fid); c += 1
            if c >= k:
                break
    log(f"selected {len(pick)}: {pick}")

    las = laspy.read(FULL / "buildings_planes.laz")
    inst = np.asarray(las.instance_id)
    styp = np.asarray(las.surf_type)
    pid = np.asarray(las.plane_id)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y)]) * FT_TO_M

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 4
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.6 * nrow))
    axes = np.atleast_2d(axes)
    s8 = np.ones((3, 3), bool)
    for k, fid in enumerate(pick):
        ax = axes[k // ncol, k % ncol]
        sel = np.nonzero(inst == fid)[0]
        rm = (styp[sel] == 1) & (pid[sel] > 0)
        P = X[sel][rm]
        x0m, y0m = P[:, 0].min(), P[:, 1].min()
        gx = ((P[:, 0] - x0m) / CELL).astype(np.int32)
        gy = ((P[:, 1] - y0m) / CELL).astype(np.int32)
        mm = np.zeros((gy.max() + 3, gx.max() + 3), bool)
        mm[gy + 1, gx + 1] = True
        m = ndimage.binary_fill_holes(ndimage.binary_closing(mm, structure=s8))
        conts = measure.find_contours(m.astype(float), 0.5)
        r = by_id[fid]
        poly = parse_wkt(r["wkt_ftus"])
        c0 = P.mean(axis=0)
        ax.scatter(P[:, 0] - c0[0], P[:, 1] - c0[1], s=0.6, c="0.75",
                   rasterized=True)
        if conts:
            cont = max(conts, key=len)
            cm = (cont[:, ::-1] - 1) * CELL
            ax.plot(cm[:, 0] + x0m - c0[0], cm[:, 1] + y0m - c0[1],
                    c="tab:blue", lw=0.8, alpha=0.8)
        pl = np.vstack([poly, poly[:1]])
        ax.plot(pl[:, 0] - c0[0], pl[:, 1] - c0[1], c="tab:red", lw=1.6)
        ax.set_title(f"id {fid} | {r['method']} | off {r['mean_offset_m']} m | "
                     f"θ={r['theta_deg']}° | ortho {r['ortho_share']}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(pick), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("Worst outlines: gray=roof pts, blue=raw mask contour, "
                 "red=final regularized outline", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "qa_worst_outlines.png", dpi=130)
    log("wrote qa_worst_outlines.png")


if __name__ == "__main__":
    main()
