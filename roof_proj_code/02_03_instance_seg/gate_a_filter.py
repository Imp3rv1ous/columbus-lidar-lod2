#!/usr/bin/env python
"""Gate A — semantic-noise removal at the instance level (before roof loop).

RandLA-Net occasionally labels roads, bridges and other clutter as building
(class 6). Reconstruction cannot reject these (bridges fit planes perfectly),
so they must be removed BEFORE the per-building roof loop, using features
that exist at the instance level:

  1. DTM from class-2 ground points (1 m grid, per-cell min, median-filtered,
     holes filled by nearest-neighbor -> terrain under buildings).
  2. Per-instance ground height = P5 of DTM in a 4 m ring around the instance
     (3D BAG convention), HAG = roof z50 - ground.
  3. Rules:
       HAG < 2.0 m                -> reject "too_low"   (roads, parking, ground clutter)
       AR >= 5 and HAG >= 2.0 m   -> keep but flag "review_elongated"
                                     (bridge/overpass candidates AND genuine
                                      row-house strips: manual review, never
                                      auto-deleted)
       otherwise                  -> keep

Outputs (output_v2/):
  instances_gateA.csv   all instances + ground/HAG/width + decision
  buildings_clean.laz   kept instances only (RGB + instance_id preserved)
  noise_removed.laz     rejected points + original id-0 noise, reason coded
  qa_gateA.png          map: kept=gray, rejected=red, review=orange + HAG hist
"""

from pathlib import Path
import csv
import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import cKDTree

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg")
PRED_DIR = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/01_semantic_randlanet/results/preds")
RGB_LAZ = BASE / "output_v2/buildings_instances_rgb.laz"
OUT = BASE / "output_v2"

FT_TO_M = 0.30480060960121924
GROUND_CLASS = 2
DTM_CELL = 1.0
CELL = 0.5
RING_R = 4.0          # m, ground sampling ring around instance (3D BAG: 4 m)
HAG_MIN = 2.0         # m, below this an instance is not a building
AR_REVIEW = 5.0       # aspect ratio: flag elongated elevated objects
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def group_slices(sorted_ids):
    u, s = np.unique(sorted_ids, return_index=True)
    e = np.append(s[1:], len(sorted_ids))
    return {a: slice(b, c) for a, b, c in zip(u, s, e)}


# ------------------------------------------------------------ global bounds
tiles = sorted(PRED_DIR.glob("*_pred.laz"))
gmin = np.array([np.inf, np.inf]); gmax = np.array([-np.inf, -np.inf])
for t in tiles:
    with laspy.open(t) as r:
        gmin = np.minimum(gmin, r.header.mins[:2])
        gmax = np.maximum(gmax, r.header.maxs[:2])
x0, y0 = gmin * FT_TO_M
nx_d = int((gmax[0] * FT_TO_M - x0) / DTM_CELL) + 2
ny_d = int((gmax[1] * FT_TO_M - y0) / DTM_CELL) + 2
log(f"DTM grid {nx_d} x {ny_d} @ {DTM_CELL} m")

# ------------------------------------------------------------ DTM from ground
dtm = np.full((ny_d, nx_d), np.inf, np.float32)
ng_total = 0
for t in tiles:
    las = laspy.read(t)
    m = np.asarray(las.classification) == GROUND_CLASS
    ng_total += int(m.sum())
    gx = (np.asarray(las.x)[m] * FT_TO_M - x0) / DTM_CELL
    gy = (np.asarray(las.y)[m] * FT_TO_M - y0) / DTM_CELL
    gz = (np.asarray(las.z)[m] * FT_TO_M).astype(np.float32)
    np.minimum.at(dtm, (gy.astype(np.int32), gx.astype(np.int32)), gz)
log(f"ground points: {ng_total:,}")
valid = np.isfinite(dtm)
log(f"DTM coverage: {valid.mean()*100:.1f}% of cells")
dtm_f = dtm.copy()
dtm_f[~valid] = np.nan
dtm_med = ndimage.median_filter(np.where(valid, dtm, 1e6), size=3)
dtm[valid] = np.where(np.abs(dtm[valid] - dtm_med[valid]) > 2.0,
                      dtm_med[valid], dtm[valid])   # kill isolated pits/spikes
_, (iy_n, ix_n) = ndimage.distance_transform_edt(~valid, return_indices=True)
dtm_filled = dtm[iy_n, ix_n]                        # nearest-neighbor infill
log("DTM filled")

# ------------------------------------------------------------ instances
las = laspy.read(RGB_LAZ)
inst = np.asarray(las.instance_id)
xm = np.asarray(las.x) * FT_TO_M
ym = np.asarray(las.y) * FT_TO_M
zm = np.asarray(las.z) * FT_TO_M
ninst = int(inst.max())
ix = ((xm - x0) / CELL).astype(np.int32)
iy = ((ym - y0) / CELL).astype(np.int32)
nx_i = int(ix.max()) + 2; ny_i = int(iy.max()) + 2

order = np.argsort(inst, kind="stable")
sl_of = group_slices(inst[order])
ring_se = None
rows = []
decision = np.zeros(ninst + 1, np.uint8)   # 0 keep, 1 reject_too_low, 2 review
r_cells = int(RING_R / CELL)
yy, xx = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
ring_se = (xx * xx + yy * yy) <= r_cells * r_cells

for fid, sl in sl_of.items():
    if fid == 0:
        continue
    idx = order[sl]
    gx, gy = ix[idx], iy[idx]
    zi = zm[idx]
    x1, x2 = gx.min(), gx.max(); y1, y2 = gy.min(), gy.max()
    pad = r_cells + 1
    ox, oy = x1 - pad, y1 - pad
    w, h = x2 - x1 + 2 * pad + 1, y2 - y1 + 2 * pad + 1
    m = np.zeros((h, w), bool)
    m[gy - oy, gx - ox] = True
    ring = ndimage.binary_dilation(m, structure=ring_se) & ~m
    ry, rx = np.nonzero(ring)
    # map 0.5 m cells -> 1 m DTM cells (clip to DTM bounds)
    dy = np.clip(((ry + oy) * CELL / DTM_CELL).astype(np.int32), 0, ny_d - 1)
    dx = np.clip(((rx + ox) * CELL / DTM_CELL).astype(np.int32), 0, nx_d - 1)
    vmask = valid[dy, dx]
    src = dtm[dy[vmask], dx[vmask]] if vmask.sum() >= 20 else dtm_filled[dy, dx]
    ground = float(np.percentile(src, 5))
    z50 = float(np.median(zi)); z95 = float(np.percentile(zi, 95))
    hag = z50 - ground
    hag95 = z95 - ground
    xyc = np.column_stack([xm[idx] - xm[idx].mean(), ym[idx] - ym[idx].mean()])
    cov = (xyc.T @ xyc) / max(len(idx) - 1, 1)
    ev = np.linalg.eigvalsh(cov)
    ar = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
    width = float(3.5 * np.sqrt(max(ev[0], 1e-9)))
    area = len(np.unique(gy.astype(np.int64) * nx_i + gx)) * CELL * CELL
    if hag < HAG_MIN:
        dec = 1
    elif ar >= AR_REVIEW:
        dec = 2
    else:
        dec = 0
    decision[fid] = dec
    rows.append((fid, len(idx), round(area, 1), round(ground, 2),
                 round(hag, 2), round(hag95, 2), round(ar, 2), round(width, 1),
                 round(float(xm[idx].mean() - x0), 1),
                 round(float(ym[idx].mean() - y0), 1),
                 {0: "keep", 1: "reject_too_low", 2: "keep_review_elongated"}[dec]))

n_rej = int((decision == 1).sum()); n_rev = int((decision == 2).sum())
log(f"instances: {ninst}  reject_too_low: {n_rej}  review_elongated: {n_rev}  "
    f"keep: {ninst - n_rej}")

with open(OUT / "instances_gateA.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["instance_id", "n_points", "area_m2", "ground_m",
                "hag_med_m", "hag_p95_m", "aspect_ratio", "width_m",
                "cx_m", "cy_m", "decision"])
    w.writerows(rows)

# ------------------------------------------------------------ split LAZ
pt_dec = decision[inst]                     # per point decision
keep_pts = (inst > 0) & (pt_dec != 1)
noise_pts = ~keep_pts
log(f"points kept {keep_pts.sum():,} ({keep_pts.mean()*100:.1f}%), removed "
    f"{noise_pts.sum():,}")

clean = laspy.LasData(las.header)
clean.points = las.points[keep_pts].copy()
clean.write(OUT / "buildings_clean.laz")

noise = laspy.LasData(las.header)
noise.points = las.points[noise_pts].copy()
noise.write(OUT / "noise_removed.laz")
log("wrote buildings_clean.laz / noise_removed.laz")

# ------------------------------------------------------------ QA image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

grid = np.zeros((ny_i, nx_i), np.uint8)     # 0 bg, 1 keep, 2 reject, 3 review
code = np.array([1, 2, 3], np.uint8)[np.clip(decision, 0, 2)]
code_full = np.zeros(ninst + 1, np.uint8); code_full[1:] = code[1:]
pos = inst > 0
np.maximum.at(grid, (iy[pos], ix[pos]), code_full[inst[pos]])
img = np.full((ny_i, nx_i, 3), 255, np.uint8)
img[grid == 1] = (185, 185, 185)
img[grid == 3] = (255, 150, 20)
img[grid == 2] = (220, 30, 30)

fig, axes = plt.subplots(1, 2, figsize=(20, 9),
                         gridspec_kw={"width_ratios": [2.6, 1]})
axes[0].imshow(img[::2, ::2], origin="lower", interpolation="nearest")
axes[0].set_title(f"Gate A: kept {ninst-n_rej} (gray) | rejected too-low "
                  f"{n_rej} (red) | review elongated {n_rev} (orange)")
hags = np.array([r[4] for r in rows])
axes[1].hist(np.clip(hags, -2, 20), bins=88, color="#4477aa")
axes[1].axvline(HAG_MIN, color="red", ls="--", label=f"HAG = {HAG_MIN} m")
axes[1].set_xlabel("height above ground, roof z50 (m)")
axes[1].set_ylabel("instances"); axes[1].legend()
axes[1].set_title("HAG distribution")
fig.tight_layout(); fig.savefig(OUT / "qa_gateA.png", dpi=150)
log("wrote qa_gateA.png")

rej = sorted([r for r in rows if r[10] == "reject_too_low"],
             key=lambda r: -r[2])[:12]
log("largest rejected (id, area m2, HAG m):")
for r in rej:
    log(f"  id {r[0]:5d}  {r[2]:8.0f} m2  hag {r[4]:5.2f}  at ({r[8]}, {r[9]})")
rev = sorted([r for r in rows if r[10] == "keep_review_elongated"],
             key=lambda r: -r[2])[:12]
log("largest review-elongated (id, area m2, HAG m, AR):")
for r in rev:
    log(f"  id {r[0]:5d}  {r[2]:8.0f} m2  hag {r[4]:5.2f}  AR {r[6]:5.1f}  at ({r[8]}, {r[9]})")
