"""Top-down PNG preview of a draft .laz: all classes + veg-only + building-only."""
import sys
from pathlib import Path

import laspy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

draft = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else draft.replace(".laz", "_preview.png")

las = laspy.read(draft)
x = np.asarray(las.x)
y = np.asarray(las.y)
c = np.asarray(las.classification).astype(int)
remap = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4}                 # ASPRS -> internal 0..4
k = np.array([remap.get(v, 0) for v in c])

n = x.shape[0]
if n > 900_000:
    rng = np.random.default_rng(0)
    sel = rng.choice(n, 900_000, replace=False)
    x, y, k = x[sel], y[sel], k[sel]

colors = ["#dddddd", "#ff8800", "#9b8060", "#1a9850", "#d7191c"]
labels = ["unlabeled", "others", "ground", "vegetation", "building"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

fig, axes = plt.subplots(1, 3, figsize=(30, 10), dpi=100)
order = np.argsort(k)
axes[0].scatter(x[order], y[order], c=k[order], cmap=cmap, norm=norm, s=0.5, linewidths=0)
axes[0].set_title("all classes")
handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=col, mec="none", ms=8) for col in colors]
axes[0].legend(handles, labels, loc="upper right", fontsize=8)

mv = k == 3
axes[1].scatter(x[mv], y[mv], c="#1a9850", s=0.5, linewidths=0)
axes[1].set_title(f"vegetation only ({mv.sum():,})")

mb = k == 4
axes[2].scatter(x[mb], y[mb], c="#d7191c", s=0.5, linewidths=0)
axes[2].set_title(f"building only ({mb.sum():,})")

for ax in axes:
    ax.set_aspect("equal")
    ax.set_facecolor("#0b1d2a")
fig.suptitle(Path(draft).stem, fontsize=14)
fig.tight_layout()
fig.savefig(out)
print("->", out)
