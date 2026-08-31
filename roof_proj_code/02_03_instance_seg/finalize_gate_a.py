#!/usr/bin/env python
"""Finalize Gate A: apply manual review verdicts on the 18 elongated
candidates, rewrite buildings_clean.laz / noise_removed.laz, update the CSV,
and redraw the audit figure.

Verdicts from visual inspection of qa_review_elongated.png (top-down crops,
height-colored): corridor bridge/wall pieces, one tree row, two power lines
rejected; the remaining 10 are genuine elongated buildings (kept, flagged)."""

from pathlib import Path
import csv
import numpy as np
import laspy

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2")
FT_TO_M = 0.30480060960121924

REJECT = {  # instance_id: reason (manual review, see qa_review_elongated.png)
    1696: "bridge_corridor", 2051: "bridge_corridor",
    2236: "bridge_corridor", 2187: "bridge_corridor",
    1780: "ramp_linear", 1370: "tree_row",
    2282: "powerline", 2277: "powerline",
}

# ------------------------------------------------------------ CSV update
with open(BASE / "instances_gateA.csv") as f:
    rows = [r for r in csv.DictReader(f)]
for r in rows:
    fid = int(r["instance_id"])
    if fid in REJECT:
        r["decision"] = f"reject_reviewed_{REJECT[fid]}"
    elif r["decision"] == "keep_review_elongated":
        r["decision"] = "keep_reviewed_ok"
with open(BASE / "instances_gateA.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------ LAZ rewrite
las = laspy.read(BASE / "buildings_instances_rgb.laz")
inst = np.asarray(las.instance_id)
rej_ids = np.array(sorted(REJECT), dtype=np.uint32)
is_rej = np.isin(inst, rej_ids)
keep_pts = (inst > 0) & ~is_rej
noise_pts = ~keep_pts

clean = laspy.LasData(las.header)
clean.points = las.points[keep_pts].copy()
clean.write(BASE / "buildings_clean.laz")
noise = laspy.LasData(las.header)
noise.points = las.points[noise_pts].copy()
noise.write(BASE / "noise_removed.laz")

nk = len({int(r['instance_id']) for r in rows
          if r['decision'].startswith('keep')})
print(f"final: kept instances {nk}, rejected {len(REJECT)} reviewed + "
      f"noise id0; points kept {keep_pts.sum():,} ({keep_pts.mean()*100:.2f}%), "
      f"removed {noise_pts.sum():,}")

# ------------------------------------------------------------ audit figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xm = np.asarray(las.x) * FT_TO_M; ym = np.asarray(las.y) * FT_TO_M
x0, y0 = xm.min(), ym.min()
CELL = 0.5
ix = ((xm - x0) / CELL).astype(np.int32)
iy = ((ym - y0) / CELL).astype(np.int32)
ny_i, nx_i = iy.max() + 2, ix.max() + 2
dec_of = np.zeros(int(inst.max()) + 1, np.uint8)  # 1 keep, 2 reject, 3 flagged
for r in rows:
    d = r["decision"]
    dec_of[int(r["instance_id"])] = 2 if d.startswith("reject") else (
        3 if d == "keep_reviewed_ok" else 1)
grid = np.zeros((ny_i, nx_i), np.uint8)
pos = inst > 0
np.maximum.at(grid, (iy[pos], ix[pos]), dec_of[inst[pos]])
img = np.full((ny_i, nx_i, 3), 255, np.uint8)
img[grid == 1] = (190, 190, 190)
img[grid == 3] = (255, 150, 20)
img[grid == 2] = (220, 30, 30)
fig, ax = plt.subplots(figsize=(15, 11))
ax.imshow(img[::2, ::2], origin="lower", interpolation="nearest")
ax.set_title(f"Gate A final: {nk} buildings kept (gray; orange = elongated, "
             f"reviewed OK) | {len(REJECT)} non-buildings removed (red: "
             f"bridges/ramps, tree row, power lines)")
fig.tight_layout(); fig.savefig(BASE / "qa_gateA.png", dpi=150)
print("wrote qa_gateA.png")
