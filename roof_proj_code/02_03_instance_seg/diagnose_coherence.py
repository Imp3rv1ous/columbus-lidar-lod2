#!/usr/bin/env python
"""Diagnose spatial coherence of v2 instances.

For every instance id in output_v2/buildings_instances.laz, rasterize its own
points (0.5 m) and run strict 8-connected labeling with NO bridging. An
instance is 'incoherent' if it decomposes into >= 2 parts that are both
substantial (>= MIN_PART m2) and separated by more than GAP m.

For the worst offenders, also checks whether the ORIGINAL occupancy grid
(all building points) contains a connecting path (thread of occupied cells),
to distinguish 'thin misclassified bridge' from 'labeling bug'.
"""

from pathlib import Path
import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import cKDTree

LAZ = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2/buildings_instances.laz")
FT_TO_M = 0.30480060960121924
CELL = 0.5
MIN_PTS_CELL = 2
MIN_PART = 10.0   # m2: a 'real' disconnected part must be at least this
GAP = 3.0         # m: parts farther apart than this are incoherent

las = laspy.read(LAZ)
xm = np.asarray(las.x) * FT_TO_M
ym = np.asarray(las.y) * FT_TO_M
zm = np.asarray(las.z) * FT_TO_M
inst = np.asarray(las.instance_id)
print(f"points {len(xm):,}, instances {inst.max()}")

x0, y0 = xm.min(), ym.min()
ix = ((xm - x0) / CELL).astype(np.int32)
iy = ((ym - y0) / CELL).astype(np.int32)
nx, ny = int(ix.max()) + 1, int(iy.max()) + 1

# original occupancy (all building points, >=2 pts per cell) for thread checks
cnt = np.zeros((ny, nx), np.uint16)
np.add.at(cnt, (iy, ix), 1)
occ_all = cnt >= MIN_PTS_CELL

order = np.argsort(inst, kind="stable")
uniq, starts = np.unique(inst[order], return_index=True)
ends = np.append(starts[1:], len(order))

s8 = np.ones((3, 3), bool)
bad = []
for u, s, e in zip(uniq, starts, ends):
    if u == 0:
        continue
    idx = order[s:e]
    gx, gy = ix[idx], iy[idx]
    x1, x2 = gx.min(), gx.max()
    y1, y2 = gy.min(), gy.max()
    m = np.zeros((y2 - y1 + 3, x2 - x1 + 3), bool)
    m[gy - y1 + 1, gx - x1 + 1] = True
    lab, k = ndimage.label(m, structure=s8)
    if k < 2:
        continue
    # part areas + pairwise gaps between substantial parts
    cys, cxs = np.nonzero(lab)
    labs = lab[cys, cxs]
    parts = []
    for j in range(1, k + 1):
        sel = labs == j
        a = sel.sum() * CELL * CELL
        parts.append((j, a, cxs[sel], cys[sel]))
    big_parts = [p for p in parts if p[1] >= MIN_PART]
    if len(big_parts) < 2:
        continue
    # max-min gap between the two biggest parts
    big_parts.sort(key=lambda p: -p[1])
    worst_gap = 0.0
    for a_i in range(len(big_parts)):
        for b_i in range(a_i + 1, len(big_parts)):
            ta = cKDTree(np.column_stack([big_parts[a_i][2], big_parts[a_i][3]]) * CELL)
            d, _ = ta.query(np.column_stack([big_parts[b_i][2], big_parts[b_i][3]]) * CELL)
            worst_gap = max(worst_gap, float(d.min()))
    if worst_gap > GAP:
        cx = (x1 + x2) / 2 * CELL
        cy = (y1 + y2) / 2 * CELL
        bad.append((int(u), len(big_parts), [round(p[1]) for p in big_parts[:5]],
                    round(worst_gap, 1), round(cx, 1), round(cy, 1),
                    (x1, x2, y1, y2)))

print(f"\nincoherent instances (>=2 parts >= {MIN_PART} m2, gap > {GAP} m): {len(bad)}")
bad.sort(key=lambda b: -b[3])
for b in bad[:15]:
    print(f"  id {b[0]:5d}: {b[1]} parts, areas {b[2]} m2, max gap {b[3]} m, "
          f"center local ({b[4]}, {b[5]})")

# thread check for the worst 3: is there a connected path in occ_all
# between the parts (i.e., a real bridge of occupied cells)?
print("\nthread check (worst 3):")
for b in bad[:3]:
    x1, x2, y1, y2 = b[6]
    pad = 40
    sl = (slice(max(y1 - pad, 0), min(y2 + pad, ny)),
          slice(max(x1 - pad, 0), min(x2 + pad, nx)))
    sub = occ_all[sl]
    lab_all, _ = ndimage.label(sub, structure=s8)
    # cells of this instance within the window
    idx = order[starts[list(uniq).index(b[0])]:ends[list(uniq).index(b[0])]]
    gy2 = iy[idx] - sl[0].start
    gx2 = ix[idx] - sl[1].start
    ok = (gy2 >= 0) & (gy2 < sub.shape[0]) & (gx2 >= 0) & (gx2 < sub.shape[1])
    comps_here = np.unique(lab_all[gy2[ok], gx2[ok]])
    comps_here = comps_here[comps_here > 0]
    print(f"  id {b[0]}: its points fall in {len(comps_here)} distinct component(s) "
          f"of the ORIGINAL occupancy -> "
          f"{'thin bridge exists (connected in raw grid)' if len(comps_here) == 1 else 'NOT connected in raw grid => labeling/post-processing bug'}")
