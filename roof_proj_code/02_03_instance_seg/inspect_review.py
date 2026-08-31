#!/usr/bin/env python
"""Render top-down crops of all Gate-A review_elongated instances for manual
classification (building vs bridge/wall/rail structure). Candidate points are
colored by height; all other building points in the window shown gray."""

from pathlib import Path
import csv
import numpy as np
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2")
FT_TO_M = 0.30480060960121924

with open(BASE / "instances_gateA.csv") as f:
    rows = [r for r in csv.DictReader(f)]
review = [r for r in rows if r["decision"] == "keep_review_elongated"]
review.sort(key=lambda r: -float(r["area_m2"]))
print(f"{len(review)} review instances")

las = laspy.read(BASE / "buildings_instances_rgb.laz")
inst = np.asarray(las.instance_id)
xm = np.asarray(las.x) * FT_TO_M
ym = np.asarray(las.y) * FT_TO_M
zm = np.asarray(las.z) * FT_TO_M
x0, y0 = xm.min(), ym.min()
xl = xm - x0; yl = ym - y0

n = len(review)
ncol = 6
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
axes = np.atleast_2d(axes)
for k, r in enumerate(review):
    ax = axes[k // ncol, k % ncol]
    fid = int(r["instance_id"])
    cx, cy = float(r["cx_m"]), float(r["cy_m"])
    half = 45.0
    w = (xl > cx - half) & (xl < cx + half) & (yl > cy - half) & (yl < cy + half)
    mine = w & (inst == fid)
    other = w & (inst != fid) & (inst > 0)
    ax.scatter(xl[other], yl[other], s=0.3, c="0.8", rasterized=True)
    zi = zm[mine]
    sc = ax.scatter(xl[mine], yl[mine], s=0.6, c=zi, cmap="viridis",
                    rasterized=True)
    zr = np.percentile(zi, 95) - np.percentile(zi, 5) if mine.sum() else 0
    ax.set_title(f"id {fid} | {float(r['area_m2']):.0f} m2 | HAG "
                 f"{r['hag_med_m']} m | AR {r['aspect_ratio']}\n"
                 f"roof z-relief {zr:.1f} m", fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for k in range(n, nrow * ncol):
    axes[k // ncol, k % ncol].axis("off")
fig.tight_layout()
fig.savefig(BASE / "qa_review_elongated.png", dpi=130)
print("wrote qa_review_elongated.png")
