#!/usr/bin/env python
"""Gate A2 — catch folded ribbon structures (interchange ramps, curved
overpasses) that pass the HAG and global-PCA-AR rules.

Global PCA aspect ratio fails on Z/bowtie-folded ribbons (locally narrow,
globally compact). Two fold-immune shape descriptors:
  solidity   = area / convex-hull area          (ribbon ~0.4-0.6, building >0.75)
  skel_elong = skeleton_len / mean_width,  mean_width = area / skeleton_len
Candidates are flagged and rendered for manual verdict (same audit-lane
pattern as Gate A)."""

from pathlib import Path
import csv
import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2")
FT_TO_M = 0.30480060960121924
CELL = 0.5

with open(BASE / "instances_gateA.csv") as f:
    rows = [r for r in csv.DictReader(f)]
dec = {int(r["instance_id"]): r["decision"] for r in rows}

las = laspy.read(BASE / "buildings_clean.laz")
inst = np.asarray(las.instance_id)
xm = np.asarray(las.x) * FT_TO_M
ym = np.asarray(las.y) * FT_TO_M
zm = np.asarray(las.z) * FT_TO_M
x0, y0 = xm.min(), ym.min()
ix = ((xm - x0) / CELL).astype(np.int32)
iy = ((ym - y0) / CELL).astype(np.int32)

order = np.argsort(inst, kind="stable")
u, s = np.unique(inst[order], return_index=True)
e = np.append(s[1:], len(order))

cand = []
feat_rows = []
for fid, a, b in zip(u, s, e):
    if fid == 0 or dec.get(int(fid), "").startswith("reject"):
        continue
    idx = order[a:b]
    gx, gy = ix[idx], iy[idx]
    x1, x2, y1, y2 = gx.min(), gx.max(), gy.min(), gy.max()
    m = np.zeros((y2 - y1 + 3, x2 - x1 + 3), bool)
    m[gy - y1 + 1, gx - x1 + 1] = True
    area = m.sum() * CELL * CELL
    if area < 80:
        continue
    cells = np.column_stack(np.nonzero(m))
    try:
        hull = ConvexHull(cells)
        solidity = float(m.sum() / hull.volume)  # 2D: volume == area
    except Exception:
        solidity = 1.0
    mf = ndimage.binary_fill_holes(m)
    # max inscribed disk radius: fold- and branch-immune local width measure
    edt = ndimage.distance_transform_edt(mf, sampling=CELL)
    width_edt = 2.0 * float(edt.max())          # ~ widest local width (m)
    elong_edt = area / (width_edt * width_edt) if width_edt > 0 else 0.0
    feat_rows.append((int(fid), round(area, 1), round(solidity, 3),
                      round(width_edt, 1), round(elong_edt, 1)))
    already_ok = dec.get(int(fid), "") == "keep_reviewed_ok"
    if already_ok:
        continue
    if int(fid) in (1, 2) or (area >= 400 and (solidity <= 0.72 or elong_edt >= 3.5)):
        zi = zm[idx]
        cand.append((int(fid), area, solidity, width_edt, elong_edt,
                     float(np.median(xm[idx] - x0)), float(np.median(ym[idx] - y0)),
                     float(np.percentile(zi, 95) - np.percentile(zi, 5))))

with open(BASE / "shape_features.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["instance_id", "area_m2", "solidity", "max_width_m", "elong_edt"])
    w.writerows(feat_rows)

cand.sort(key=lambda c: -c[1])
print(f"flagged candidates: {len(cand)}")
for c in cand:
    print(f"  id {c[0]:5d}  {c[1]:7.0f} m2  solidity {c[2]:.2f}  width {c[3]:4.1f} m"
          f"  skel_elong {c[4]:5.1f}  at ({c[5]:.0f}, {c[6]:.0f})  z-relief {c[7]:.1f}")

# render
n = len(cand)
if n:
    ncol = min(6, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes)
    xl, yl = xm - x0, ym - y0
    for k, c in enumerate(cand):
        ax = axes[k // ncol, k % ncol]
        fid, cx, cy = c[0], c[5], c[6]
        half = max(60.0, np.sqrt(c[1]))
        w = (xl > cx - half) & (xl < cx + half) & (yl > cy - half) & (yl < cy + half)
        mine = w & (inst == fid)
        other = w & (inst != fid) & (inst > 0)
        ax.scatter(xl[other], yl[other], s=0.3, c="0.85", rasterized=True)
        ax.scatter(xl[mine], yl[mine], s=0.6, c=zm[mine], cmap="viridis",
                   rasterized=True)
        ax.set_title(f"id {fid} | {c[1]:.0f} m2 | sol {c[2]:.2f} | w {c[3]:.0f} m\n"
                     f"skel_elong {c[4]:.1f} | z-relief {c[7]:.1f} m", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(BASE / "qa_shape_candidates.png", dpi=130)
    print("wrote qa_shape_candidates.png")
