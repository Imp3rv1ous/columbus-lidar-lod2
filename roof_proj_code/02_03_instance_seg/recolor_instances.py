#!/usr/bin/env python
"""Bake collision-free per-instance RGB into the v2 LAZ for reliable QA.

Why: instance_id is assigned by descending area, so two similar-sized
buildings anywhere in the scene get numerically adjacent ids; CloudCompare's
continuous color ramp then renders them in indistinguishable colors, which
looks like "distant buildings share one instance id". The data itself is
coherent (verified by diagnose_coherence.py: 0 instances with far-apart
parts). This script writes a point-format-7 LAZ with true RGB per instance
(golden-angle hue stepping + S/V cycling, deduplicated), plus a `color_id`
extra dim (pseudo-random permutation of instance_id) for SF-based coloring.
Also regenerates the QA PNGs with the same palette so the images and the
point cloud agree.
"""

import colorsys
from pathlib import Path

import numpy as np
import laspy

IN_LAZ = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2/buildings_instances.laz")
OUT_LAZ = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/02_03_instance_seg/output_v2/buildings_instances_rgb.laz")
OUT_DIR = IN_LAZ.parent
FT_TO_M = 0.30480060960121924
CELL = 0.5
CHECK_FT = (1841781.790039, 711006.650024)


def make_palette(n):
    """n visually distinct RGB triples (uint8); id 0 = gray for noise."""
    sats = [0.95, 0.70, 0.55]
    vals = [0.95, 0.75, 0.60, 0.85]
    cols = np.zeros((n + 1, 3), np.uint8)
    cols[0] = (140, 140, 140)
    seen = set()
    phi = 0.6180339887498949
    for i in range(1, n + 1):
        h = (i * phi) % 1.0
        s = sats[i % 3]
        v = vals[i % 4]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        c = (int(r * 255), int(g * 255), int(b * 255))
        while c in seen:  # nudge value until unique
            v = v - 0.03 if v > 0.2 else v + 0.35
            r, g, b = colorsys.hsv_to_rgb(h, s, min(v, 1.0))
            c = (int(r * 255), int(g * 255), int(b * 255))
        seen.add(c)
        cols[i] = c
    return cols


las = laspy.read(IN_LAZ)
inst = np.asarray(las.instance_id)
src_tile = np.asarray(las.src_tile)
ninst = int(inst.max())
print(f"{len(inst):,} points, {ninst} instances")

pal = make_palette(ninst)
print(f"palette: {len(np.unique(pal.view([('', pal.dtype)] * 3)))} unique colors "
      f"for {ninst + 1} ids")

header = laspy.LasHeader(point_format=7, version="1.4")  # 6 + RGB
header.scales = las.header.scales
header.offsets = las.header.offsets
for vlr in las.header.vlrs:
    header.vlrs.append(vlr)
header.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.uint32))
header.add_extra_dim(laspy.ExtraBytesParams(name="src_tile", type=np.uint8))
header.add_extra_dim(laspy.ExtraBytesParams(name="color_id", type=np.uint16))

out = laspy.LasData(header)
out.x = np.asarray(las.x)
out.y = np.asarray(las.y)
out.z = np.asarray(las.z)
out.classification = np.asarray(las.classification)
out.instance_id = inst
out.src_tile = src_tile
# pseudo-random permutation for SF display (Knuth multiplicative hash)
out.color_id = ((inst.astype(np.uint64) * 2654435761) % 65521).astype(np.uint16)
rgb = pal[inst]
out.red = rgb[:, 0].astype(np.uint16) * 257
out.green = rgb[:, 1].astype(np.uint16) * 257
out.blue = rgb[:, 2].astype(np.uint16) * 257
out.write(OUT_LAZ)
print(f"wrote {OUT_LAZ}")

# ------------------------------------------------------------- QA PNGs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xm = np.asarray(las.x) * FT_TO_M
ym = np.asarray(las.y) * FT_TO_M
x0, y0 = xm.min(), ym.min()
ix = ((xm - x0) / CELL).astype(np.int32)
iy = ((ym - y0) / CELL).astype(np.int32)
nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
grid = np.zeros((ny, nx), np.uint32)
grid[iy, ix] = inst  # last write wins; fine for visualization
img = np.full((ny, nx, 3), 255, np.uint8)
occ = grid > 0
img[occ] = pal[grid[occ]]

fig, ax = plt.subplots(figsize=(14, 11))
ax.imshow(img[::2, ::2], origin="lower", interpolation="nearest")
ax.set_title(f"v2 instances, collision-free palette ({ninst} instances)")
fig.tight_layout(); fig.savefig(OUT_DIR / "qa_overview_rgb.png", dpi=160)
plt.close(fig)

cxq = CHECK_FT[0] * FT_TO_M - x0
cyq = CHECK_FT[1] * FT_TO_M - y0
half = 120.0
x1, x2 = int((cxq - half) / CELL), int((cxq + half) / CELL)
y1, y2 = int((cyq - half) / CELL), int((cyq + half) / CELL)
fig, ax = plt.subplots(figsize=(11, 11))
ax.imshow(img[max(y1, 0):y2, max(x1, 0):x2], origin="lower",
          interpolation="nearest",
          extent=[x1 * CELL, x2 * CELL, y1 * CELL, y2 * CELL])
ax.set_title("check point, collision-free palette")
ax.set_xlabel("x (m, local)"); ax.set_ylabel("y (m, local)")
fig.tight_layout(); fig.savefig(OUT_DIR / "qa_checkpoint_rgb.png", dpi=160)
plt.close(fig)
print("wrote QA images")
