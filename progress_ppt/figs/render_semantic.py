"""Render RandLA-Net semantic segmentation figures for the progress PPT.

Outputs:
  sem_overview.png  - full study area, 1-m grid, colored by class (priority render)
  sem_zoom.png      - 400x400 m window around the user's check point, point-level scatter
Local coordinate frame matches instance_segmentation_v2.py: meters, origin at the
min x/y of building-class points across the 19 tiles.
"""
import glob
import numpy as np
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FT_TO_M = 0.30480060960121924
CHECK_FT = (1841781.790039, 711006.650024)
PRED_DIR = "/Users/baioqile/Documents/roof_proj/roof_proj_code/01_semantic_randlanet/results/preds"
OUT_DIR = "./_build"

CLASS_COLORS = {
    1: (0.82, 0.80, 0.78),   # others - warm gray
    2: (0.92, 0.90, 0.87),   # ground - light beige
    5: (0.55, 0.72, 0.47),   # vegetation - green
    6: (0.73, 0.09, 0.09),   # building - scarlet
}
CLASS_NAMES = {1: "other", 2: "ground", 5: "vegetation", 6: "building"}

tiles = sorted(glob.glob(PRED_DIR + "/*.laz"))
print(f"{len(tiles)} tiles")

xs_all, ys_all, cl_all = [], [], []
for t in tiles:
    las = laspy.read(t)
    c = np.asarray(las.classification)
    x = np.asarray(las.x) * FT_TO_M
    y = np.asarray(las.y) * FT_TO_M
    keep = np.arange(0, len(c), 4)  # decimate 1/4 for overview grid
    xs_all.append(x[keep].astype(np.float64))
    ys_all.append(y[keep].astype(np.float64))
    cl_all.append(c[keep])
    print("read", t.split("/")[-1], len(c))

x = np.concatenate(xs_all); y = np.concatenate(ys_all); c = np.concatenate(cl_all)
del xs_all, ys_all, cl_all

mb = c == 6
x0, y0 = x[mb].min(), y[mb].min()
xl, yl = x - x0, y - y0
print("local extent", xl.min(), xl.max(), yl.min(), yl.max())

# ---- overview: 1-m grid, priority render (building > veg > ground > other)
GRID = 1.0
gx0, gy0 = xl.min(), yl.min()
nx = int(np.ceil((xl.max() - gx0) / GRID)) + 1
ny = int(np.ceil((yl.max() - gy0) / GRID)) + 1
img = np.ones((ny, nx, 3))
prio = np.zeros((ny, nx), np.int8)
order = [(1, 1), (2, 2), (5, 3), (6, 4)]
ix = ((xl - gx0) / GRID).astype(np.int32)
iy = ((yl - gy0) / GRID).astype(np.int32)
for cls, p in order:
    m = c == cls
    if not m.any():
        continue
    sel_iy, sel_ix = iy[m], ix[m]
    upd = np.zeros((ny, nx), bool)
    upd[sel_iy, sel_ix] = True
    upd &= prio < p
    img[upd] = CLASS_COLORS[cls]
    prio[upd] = p

fig, ax = plt.subplots(figsize=(11, 8.2), dpi=200)
ax.imshow(img, origin="lower", extent=[gx0, gx0 + nx * GRID, gy0, gy0 + ny * GRID],
          interpolation="nearest")
ax.set_xlabel("x (m, local)", fontsize=11)
ax.set_ylabel("y (m, local)", fontsize=11)
ax.tick_params(labelsize=10)
handles = [Patch(facecolor=CLASS_COLORS[k], label=CLASS_NAMES[k]) for k in (6, 5, 2, 1)]
ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT_DIR + "/sem_overview.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("overview saved")

# ---- zoom: 400x400 m around check point, full-density scatter
cx = CHECK_FT[0] * FT_TO_M - x0
cy = CHECK_FT[1] * FT_TO_M - y0
half = 200.0
print("check local", cx, cy)

zx, zy, zc = [], [], []
for t in tiles:
    las = laspy.read(t)
    xt = np.asarray(las.x) * FT_TO_M - x0
    yt = np.asarray(las.y) * FT_TO_M - y0
    m = (np.abs(xt - cx) <= half) & (np.abs(yt - cy) <= half)
    if not m.any():
        continue
    zx.append(xt[m]); zy.append(yt[m]); zc.append(np.asarray(las.classification)[m])
zx = np.concatenate(zx); zy = np.concatenate(zy); zc = np.concatenate(zc)
print("zoom points", len(zx))
step = max(1, len(zx) // 1_500_000)
zx, zy, zc = zx[::step], zy[::step], zc[::step]

fig, ax = plt.subplots(figsize=(8.6, 8.6), dpi=200)
for cls in (2, 1, 5, 6):  # draw building last so it sits on top
    m = zc == cls
    ax.scatter(zx[m], zy[m], s=0.03, c=[CLASS_COLORS[cls]], marker=".",
               linewidths=0, rasterized=True)
ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
ax.set_aspect("equal")
ax.set_xlabel("x (m, local)", fontsize=11)
ax.set_ylabel("y (m, local)", fontsize=11)
ax.tick_params(labelsize=10)
handles = [Patch(facecolor=CLASS_COLORS[k], label=CLASS_NAMES[k]) for k in (6, 5, 2, 1)]
ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT_DIR + "/sem_zoom.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("zoom saved")
