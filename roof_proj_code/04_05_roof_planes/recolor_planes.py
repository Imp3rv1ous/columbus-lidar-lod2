#!/usr/bin/env python
"""Bake per-plane collision-free RGB into buildings_planes_rgb.laz
(walls darkened 65%, unassigned gray), plus plane_color_id dim."""
import colorsys
from pathlib import Path
import numpy as np
import laspy

OUT = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/04_05_roof_planes/output_full")
las = laspy.read(OUT / "buildings_planes.laz")
pid = np.asarray(las.plane_id)
styp = np.asarray(las.surf_type)
nmax = int(pid.max())
sats = [0.95, 0.7, 0.55]; vals = [0.95, 0.75, 0.6, 0.85]
pal = np.zeros((nmax + 1, 3), np.uint8); pal[0] = (200, 200, 200)
phi = 0.6180339887498949; seen = set()
for i in range(1, nmax + 1):
    h = (i * phi) % 1.0; s = sats[i % 3]; v = vals[i % 4]
    c = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
    while c in seen:
        v = v - 0.03 if v > 0.2 else v + 0.35
        c = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, min(v, 1.0)))
    seen.add(c); pal[i] = c
rgb = pal[pid]
wall = styp == 2
rgb[wall] = (rgb[wall] * 0.35).astype(np.uint8)
if "plane_color_id" not in las.point_format.dimension_names:
    las.add_extra_dim(laspy.ExtraBytesParams(name="plane_color_id", type=np.uint16))
las.plane_color_id = ((pid.astype(np.uint64) * 2654435761) % 65521).astype(np.uint16)
las.red = rgb[:, 0].astype(np.uint16) * 257
las.green = rgb[:, 1].astype(np.uint16) * 257
las.blue = rgb[:, 2].astype(np.uint16) * 257
las.write(OUT / "buildings_planes_rgb.laz")
print(f"wrote buildings_planes_rgb.laz: {len(seen)+1} unique colors, {nmax} planes")
