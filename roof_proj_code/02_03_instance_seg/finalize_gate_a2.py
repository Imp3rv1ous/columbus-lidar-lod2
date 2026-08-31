#!/usr/bin/env python
"""Finalize Gate A2: add the 5 ribbon/bridge rejects found by shape review
(solidity + EDT-width flags, verdicts from qa_shape_candidates.png and
qa_three_checks.png profiles) on top of the 8 Gate-A rejects, and rewrite
the clean/noise LAZ, CSV and audit figure."""

from pathlib import Path
import csv
import numpy as np
import laspy

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2")
FT_TO_M = 0.30480060960121924

NEW_REJECT = {
    2: "ramp_zigzag",      # 5549 m2 folded ribbon, rail corridor (user's find)
    32: "ramp_bowtie",     # 1367 m2 hourglass with continuous grade (user's find)
    90: "ramp_bowtie",     # 586 m2 hourglass, east interchange
    57: "bridge_deck",     # arched profile over corridor
    53: "bridge_deck",     # flat deck + parapet over corridor
}

with open(BASE / "instances_gateA.csv") as f:
    rows = [r for r in csv.DictReader(f)]
for r in rows:
    fid = int(r["instance_id"])
    if fid in NEW_REJECT:
        r["decision"] = f"reject_reviewed_{NEW_REJECT[fid]}"
with open(BASE / "instances_gateA.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

reject_all = {int(r["instance_id"]) for r in rows
              if r["decision"].startswith("reject")}
print(f"total rejected instances: {len(reject_all)}")

las = laspy.read(BASE / "buildings_instances_rgb.laz")
inst = np.asarray(las.instance_id)
is_rej = np.isin(inst, np.array(sorted(reject_all), dtype=np.uint32))
keep_pts = (inst > 0) & ~is_rej
clean = laspy.LasData(las.header)
clean.points = las.points[keep_pts].copy()
clean.write(BASE / "buildings_clean.laz")
noise = laspy.LasData(las.header)
noise.points = las.points[~keep_pts].copy()
noise.write(BASE / "noise_removed.laz")
nk = len({int(r["instance_id"]) for r in rows
          if r["decision"].startswith("keep")})
print(f"kept instances {nk}; points kept {keep_pts.sum():,} "
      f"({keep_pts.mean()*100:.2f}%), removed {(~keep_pts).sum():,}")

# audit figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
xm = np.asarray(las.x) * FT_TO_M; ym = np.asarray(las.y) * FT_TO_M
x0, y0 = xm.min(), ym.min()
CELL = 0.5
ix = ((xm - x0) / CELL).astype(np.int32)
iy = ((ym - y0) / CELL).astype(np.int32)
ny_i, nx_i = iy.max() + 2, ix.max() + 2
dec_of = np.zeros(int(inst.max()) + 1, np.uint8)
for r in rows:
    dec_of[int(r["instance_id"])] = 2 if r["decision"].startswith("reject") else 1
grid = np.zeros((ny_i, nx_i), np.uint8)
pos = inst > 0
np.maximum.at(grid, (iy[pos], ix[pos]), dec_of[inst[pos]])
img = np.full((ny_i, nx_i, 3), 255, np.uint8)
img[grid == 1] = (190, 190, 190)
img[grid == 2] = (220, 30, 30)
fig, ax = plt.subplots(figsize=(15, 11))
ax.imshow(img[::2, ::2], origin="lower", interpolation="nearest")
ax.set_title(f"Gate A final: {nk} buildings (gray) | {len(reject_all)} "
             f"non-buildings removed (red: bridges, ramps, rail structures, "
             f"tree rows, power lines)")
fig.tight_layout(); fig.savefig(BASE / "qa_gateA.png", dpi=150)
print("wrote qa_gateA.png")
