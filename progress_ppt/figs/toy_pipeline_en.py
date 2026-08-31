#!/usr/bin/env python3
"""玩具小区演示：用 instance_segmentation_v2.py / gate_a_filter.py 里
一模一样的操作，跑一个 25 m x 18 m 的合成场景，每一站输出前后对比图。

场景（都是"被标成建筑"的点，除了 ground）：
  A  人字顶住宅 10x8 m（屋檐 4.6 / 屋脊 6.0），屋顶有个 1.5x1.5 m 的洞
  B  平顶房 9x7 m，z=4.0；与 A 隔 2 m 侧院
  桥 树枝错点带 2x1.5 m，z=2.8 —— 把 A 和 B 连起来（要靠分水岭拆）
  门廊 3x2.5 m，z=3.2，距 A 1 m —— 演示碎片吸收
  C  独立棚屋 5x4.6 m，z=3.0 —— 腐蚀后无核，保持整块
  停车场 6x5 m，z=0.15 —— 贴地错标，留给 Gate A 的离地高度规则杀
  零星错点 5 个孤点（每格 1 个）—— 被 >=2 占据规则杀
  杂物 2x1.5 m，z=1.5，远离一切 —— 被 <20 m^2 丢渣规则杀
  屋檐稀疏点 A 左侧一条 0.5 m 窄带，每格只有 1 点 —— 孤儿点回挂演示
  ground 全场类别 2 地面点，z ~ 0.02x —— 建 DTM 算离地高度
"""
import json
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

# ---------------- 中文字体 ----------------
avail = {f.name for f in font_manager.fontManager.ttflist}
for cand in ["Helvetica", "Arial", "DejaVu Sans"]:
    if cand in avail:
        plt.rcParams["font.sans-serif"] = [cand]
        print("font:", cand)
        break
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
FIG = "figs_en/"

# ---------------- 造点 ----------------
def patch(x1, x2, y1, y2, dens, zfun):
    n = max(1, int((x2 - x1) * (y2 - y1) * dens))
    x = rng.uniform(x1, x2, n); y = rng.uniform(y1, y2, n)
    z = zfun(x, y) + rng.normal(0, 0.03, n)
    return x, y, z

parts = {}
parts["A"] = patch(2, 12, 2, 10, 24, lambda x, y: 6.0 - 0.35 * np.abs(y - 6))
parts["porch"] = patch(2, 5, 11, 13.5, 24, lambda x, y: 3.2 + 0 * x)
parts["bridge"] = patch(12, 14, 5.5, 7.0, 14, lambda x, y: 2.8 + 0 * x)
parts["B"] = patch(14, 23, 2, 9, 24, lambda x, y: 4.0 + 0 * x)
parts["C"] = patch(17, 22, 11.5, 16.1, 24, lambda x, y: 3.0 + 0 * x)
parts["parking"] = patch(8.5, 14.5, 12.5, 17.5, 24, lambda x, y: 0.15 + 0 * x)
parts["junk"] = patch(0.5, 2.5, 15.5, 17.0, 24, lambda x, y: 1.5 + 0 * x)
# A 屋顶挖洞
xa, ya, za = parts["A"]
hole = (xa > 5) & (xa < 6.5) & (ya > 4) & (ya < 5.5)
parts["A"] = (xa[~hole], ya[~hole], za[~hole])
# 零星错点：5 个孤点（不同格子）
sx = np.array([12.3, 13.1, 13.7, 12.6, 13.4])
sy = np.array([2.6, 3.1, 3.8, 4.4, 2.9])
parts["stray"] = (sx, sy, np.full(5, 3.0))
# 屋檐稀疏带（孤儿点）：每格约 1 点
ox, oy, oz = patch(1.45, 1.95, 3, 9, 4, lambda x, y: 4.5 + 0 * x)
parts["orphan"] = (ox, oy, oz)

x = np.concatenate([p[0] for p in parts.values()])
y = np.concatenate([p[1] for p in parts.values()])
z = np.concatenate([p[2] for p in parts.values()])
src = np.concatenate([np.full(len(p[0]), k) for k, p in parts.items()])
print("building-labeled pts:", len(x))

# ground（类别 2）
gmask_boxes = [(2, 12, 2, 10), (14, 23, 2, 9), (17, 22, 11.5, 16.1),
               (2, 5, 11, 13.5), (8.5, 14.5, 12.5, 17.5)]
gx = rng.uniform(0, 25, 2200); gy = rng.uniform(0, 18, 2200)
keep = np.ones(len(gx), bool)
for (a, b, c, d) in gmask_boxes:
    keep &= ~((gx > a) & (gx < b) & (gy > c) & (gy < d))
gx, gy = gx[keep], gy[keep]
gz = 0.02 * gx + rng.normal(0, 0.02, len(gx))
print("ground pts:", len(gx))

COL = {"A": "#4477aa", "B": "#66ccee", "C": "#228833", "porch": "#aa3377",
       "parking": "#ccbb44", "bridge": "#cc3311", "stray": "#cc3311",
       "junk": "#777777", "orphan": "#ee7733"}
NAME = {"A": "house A (gable roof)", "B": "flat-roof house B", "C": "shed C",
        "porch": "porch fragment", "parking": "mislabeled parking lot", "bridge": "tree-branch bridge",
        "stray": "stray points", "junk": "junk cluster", "orphan": "sparse eave points"}

# ============ F1 场景总览 ============
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6),
                         gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
ax.scatter(gx, gy, s=1.5, c="#dddddd", label="ground points (class 2)")
for k in parts:
    px, py, _ = parts[k]
    ax.scatter(px, py, s=3 if k not in ("stray",) else 30,
               c=COL[k], label=NAME[k],
               marker="o" if k != "stray" else "x")
ax.set_aspect("equal"); ax.set_xlim(0, 25); ax.set_ylim(0, 18)
ax.set_title("Toy scene, top view (colors = true identity; the algorithm only sees \"building\" points)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=9,
          markerscale=2)
ax = axes[1]
sl = (y > 5.9) & (y < 6.5)
ax.scatter(x[sl], z[sl], s=4, c=[COL[s] for s in src[sl]])
gsl = (gy > 5.9) & (gy < 6.5)
ax.scatter(gx[gsl], gz[gsl], s=3, c="#bbbbbb")
ax.set_title(r"Side profile along y $\approx$ 6.2 m", fontsize=11)
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
ax.set_xlim(0, 25)
fig.tight_layout(); fig.savefig(FIG + "toy_f1_scene.png", dpi=150)
plt.close(fig)

# ============ 站 3：栅格化（与 v2 完全一致） ============
CELL = 0.5
MIN_PTS_CELL = 2
x0, y0 = 0.0, 0.0
ix = ((x - x0) / CELL).astype(np.int32)
iy = ((y - y0) / CELL).astype(np.int32)
nx, ny = int(ix.max()) + 2, int(iy.max()) + 2

cnt = np.zeros((ny, nx), np.uint16)
np.add.at(cnt, (iy, ix), 1)
occ = cnt >= MIN_PTS_CELL
zg = np.full((ny, nx), -1e9, np.float32)
np.maximum.at(zg, (iy, ix), z.astype(np.float32))
print("occupied cells:", int(occ.sum()),
      "dropped speckle cells:", int(((cnt > 0) & ~occ).sum()))

ext = [0, nx * CELL, 0, ny * CELL]

def show_grid(ax, img, cmap, title, vmin=None, vmax=None):
    ax.imshow(img, origin="lower", extent=ext, cmap=cmap,
              interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

# F2：计数放大镜 + 占据图
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                         gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
zx1, zx2, zy1, zy2 = 11.0, 15.0, 2.0, 5.5   # 放大窗：零星错点区
sel = (x > zx1) & (x < zx2) & (y > zy1) & (y < zy2)
ax.scatter(x[sel], y[sel], s=12, c=[COL[s] for s in src[sel]])
for gx_ in np.arange(zx1, zx2 + 0.01, CELL):
    ax.axvline(gx_, color="0.8", lw=0.6)
for gy_ in np.arange(zy1, zy2 + 0.01, CELL):
    ax.axhline(gy_, color="0.8", lw=0.6)
for cx in range(int(zx1 / CELL), int(zx2 / CELL)):
    for cy in range(int(zy1 / CELL), int(zy2 / CELL)):
        c = cnt[cy, cx]
        if c > 0:
            ax.text((cx + 0.5) * CELL, (cy + 0.5) * CELL, str(c),
                    ha="center", va="center", fontsize=7,
                    color="k" if c >= MIN_PTS_CELL else "#cc3311",
                    fontweight="bold" if c < MIN_PTS_CELL else "normal")
ax.set_aspect("equal"); ax.set_xlim(zx1, zx2); ax.set_ylim(zy1, zy2)
ax.set_title("Zoom: points counted per 0.5 m cell\n(red digit = only 1 point, not occupied)")
ax.set_xlabel("x (m)")
ax = axes[1]
img = np.zeros((*occ.shape, 3))
img[...] = 1.0
img[(cnt > 0) & ~occ] = (1.0, 0.75, 0.7)
img[occ] = (0.25, 0.4, 0.65)
ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
ax.set_title("Occupancy: dark blue = cells with $\\geq$ 2 points; light red = 1-point cells, set aside\n(all 5 stray points fall in light red, so no single-point bridge can form)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout(); fig.savefig(FIG + "toy_f2_raster.png", dpi=150)
plt.close(fig)

# ============ 站 4：连通域 ============
s8 = np.ones((3, 3), bool)
comps0, nc0 = ndimage.label(occ, structure=s8)
print("raw components:", nc0)
sizes0 = np.bincount(comps0.ravel()); sizes0[0] = 0
order0 = np.argsort(-sizes0)

# F3：连通域上色（注意 A+桥+B 同色）
fig, ax = plt.subplots(figsize=(8.6, 6.2))
lut = np.array([[1, 1, 1],
                [0.27, 0.47, 0.67],   # 最大块（A+桥+B）
                [0.8, 0.73, 0.27],
                [0.13, 0.53, 0.2],
                [0.67, 0.2, 0.47],
                [0.5, 0.5, 0.5],
                [0.9, 0.6, 0.2], [0.3, 0.7, 0.7]])
rank = np.zeros(nc0 + 1, int)
for r, cid in enumerate(order0[:nc0], 1):
    if cid > 0:
        rank[cid] = min(r, 7)
ax.imshow(lut[rank[comps0]], origin="lower", extent=ext,
          interpolation="nearest")
ax.set_title(f"8-connected labeling: {nc0} blocks, but the big left block is really A + tree bridge + B")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.annotate("A + bridge + B\nwelded into one!", xy=(13, 6.2), xytext=(15.5, 12),
            arrowprops=dict(arrowstyle="->", color="k"), fontsize=11)
fig.tight_layout(); fig.savefig(FIG + "toy_f3_components.png", dpi=150)
plt.close(fig)

# ============ 站 5：闭运算对照 + 逐块补洞 ============
closed = ndimage.binary_closing(occ, structure=np.ones((5, 5), bool))
# v2 真正做的：逐块 fill_holes
filled = np.zeros_like(occ)
objs = ndimage.find_objects(comps0)
for ci in range(1, nc0 + 1):
    slc = objs[ci - 1]
    m = comps0[slc] == ci
    mf = ndimage.binary_fill_holes(m)
    filled[slc] |= mf

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
show_grid(axes[0], occ, "Blues", "Original occupancy: a hole in A's roof, a 2 m side-yard gap between A and B")
axes[0].annotate("hole", xy=(5.75, 4.75), xytext=(4, 12.5),
                 arrowprops=dict(arrowstyle="->"), fontsize=11)
axes[0].annotate("side-yard gap", xy=(13, 3.5), xytext=(16, 12.6),
                 arrowprops=dict(arrowstyle="->"), fontsize=11)
show_grid(axes[1], closed, "Reds",
          "With morphological closing (the v1 mistake):\nhole filled, but the side-yard gap is welded shut")
show_grid(axes[2], filled, "Greens",
          "v2: per-block fill_holes:\nhole filled, gap preserved")
fig.tight_layout(); fig.savefig(FIG + "toy_f4_closing.png", dpi=150)
plt.close(fig)

# ============ 站 6+7：腐蚀取核 + 分水岭（对最大块） ============
ERODE_R = 1.5
MIN_CORE_AREA = 6.0
def disk(r_cells):
    yy, xx = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
    return (xx * xx + yy * yy) <= r_cells * r_cells
selem = disk(int(round(ERODE_R / CELL)))
min_core_cells = int(MIN_CORE_AREA / (CELL * CELL))

labels2 = np.zeros((ny, nx), np.int32)
next_id = 1
core_vis = np.zeros((ny, nx), np.int32)   # 仅供画图
split_info = []
for ci in range(1, nc0 + 1):
    slc = objs[ci - 1]
    m = comps0[slc] == ci
    mf = ndimage.binary_fill_holes(m)
    core = ndimage.binary_erosion(mf, structure=selem) & m
    lab_c, k = ndimage.label(core, structure=s8)
    valid = []
    if k >= 2:
        szs = np.bincount(lab_c.ravel())
        valid = [j for j in range(1, k + 1) if szs[j] >= min_core_cells]
    cw = core_vis[slc]
    cw[core] = ci
    if len(valid) >= 2:
        markers = np.zeros(m.shape, np.int32)
        for r, j in enumerate(valid, 1):
            markers[lab_c == j] = r
        ws = watershed(-zg[slc], markers=markers, mask=mf, connectivity=2)
        tgt = labels2[slc]
        good = m & (ws > 0)
        tgt[good] = ws[good] + next_id - 1
        split_info.append((ci, len(valid)))
        next_id += len(valid)
    else:
        labels2[slc][m] = next_id
        next_id += 1
nlab = next_id - 1
print("labels after watershed:", nlab, "splits:", split_info)

# F5：腐蚀取核
fig, ax = plt.subplots(figsize=(8.6, 6.2))
base = np.zeros((*occ.shape, 3)); base[...] = 1.0
base[filled] = (0.85, 0.88, 0.93)
base[core_vis > 0] = (0.15, 0.3, 0.55)
ax.imshow(base, origin="lower", extent=ext, interpolation="nearest")
ax.set_title("Erosion cores (shrink every block by 1.5 m): light = original block, dark = core\n"
             "big block keeps 2 valid cores $\\Rightarrow$ two buildings inside; all others $\\leq$ 1 core $\\Rightarrow$ keep whole")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.text(7, 6, "core 1", color="white", fontsize=13, fontweight="bold",
        ha="center", va="center")
ax.text(18.3, 5.5, "core 2", color="white", fontsize=13, fontweight="bold",
        ha="center", va="center")
ax.annotate("bridge eroded away, cores separated", xy=(13.0, 5.9), xytext=(9.5, 0.6),
            arrowprops=dict(arrowstyle="->", color="#cc3311"),
            color="#cc3311", fontsize=11)
fig.tight_layout(); fig.savefig(FIG + "toy_f5_cores.png", dpi=150)
plt.close(fig)

# F6：分水岭（地形 + 结果 + 剖面）
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0),
                         gridspec_kw={"width_ratios": [1, 1, 1.1]})
zshow = np.where(zg > -1e8, zg, np.nan)
im = axes[0].imshow(zshow, origin="lower", extent=ext, cmap="viridis",
                    interpolation="nearest")
axes[0].set_title("Max height per cell $z_{max}$:\nroofs bright, tree bridge dark (a valley)")
plt.colorbar(im, ax=axes[0], shrink=0.8, label="z (m)")
axes[0].set_xlabel("x (m)")
lut2 = np.array([[1, 1, 1], [0.27, 0.47, 0.67], [0.4, 0.8, 0.87],
                 [0.8, 0.73, 0.27], [0.13, 0.53, 0.2], [0.67, 0.2, 0.47],
                 [0.5, 0.5, 0.5], [0.9, 0.6, 0.2], [0.3, 0.7, 0.7],
                 [0.6, 0.3, 0.6]])
axes[1].imshow(lut2[np.clip(labels2, 0, 9)], origin="lower", extent=ext,
               interpolation="nearest")
axes[1].set_title("After marker watershed:\nthe block is cut into A and B along the valley")
axes[1].set_xlabel("x (m)")
row = int(6.25 / CELL)
prof_x = (np.arange(nx) + 0.5) * CELL
prof_z = np.where(zg[row] > -1e8, zg[row], np.nan)
axes[2].plot(prof_x, prof_z, "-o", ms=2.5, lw=1.2, color="#4477aa")
lab_row = labels2[row]
bnd = None
for cxx in range(1, nx):
    if lab_row[cxx] > 0 and lab_row[cxx - 1] > 0 and lab_row[cxx] != lab_row[cxx - 1]:
        bnd = (cxx) * CELL
        break
if bnd is None:
    on = np.nonzero(lab_row > 0)[0]
    changes = on[np.nonzero(np.diff(lab_row[on]) != 0)[0]] + 1
    if len(changes):
        bnd = changes[0] * CELL
if bnd is not None:
    axes[2].axvline(bnd, color="#cc3311", ls="--", lw=2)
    axes[2].text(bnd + 0.3, 5.6, "watershed dam", color="#cc3311", fontsize=10)
axes[2].set_title(r"$z_{max}$ profile along y $\approx$ 6.2 m:" + "\nthe dam falls in the tree-bridge valley")
axes[2].set_xlabel("x (m)"); axes[2].set_ylabel("z (m)")
fig.tight_layout(); fig.savefig(FIG + "toy_f6_watershed.png", dpi=150)
plt.close(fig)

# ============ 站 8：碎片吸收（与 v2 相同逻辑） ============
NOISE_AREA = 20.0
BIG_MIN_AREA = 50.0
ABSORB_MAX_AREA = 20.0
ABSORB_DIST = 3.0
ABSORB_MAX_DZ = 3.0

lab_p = labels2[iy, ix]
area = np.zeros(nlab + 1)
for c in range(1, nlab + 1):
    area[c] = (labels2 == c).sum() * CELL * CELL
z50 = np.zeros(nlab + 1)
for c in range(1, nlab + 1):
    pts = lab_p == c
    if pts.any():
        z50[c] = np.median(z[pts])
cys, cxs = np.nonzero(labels2 > 0)
clab = labels2[cys, cxs]
big_ids = np.nonzero(area >= BIG_MIN_AREA)[0]
big_sel = np.isin(clab, big_ids)
tree = cKDTree(np.column_stack([cxs[big_sel] * CELL, cys[big_sel] * CELL]))
big_l = clab[big_sel]
remap = np.arange(nlab + 1)
absorb_log = []
for c in range(1, nlab + 1):
    if not (0 < area[c] < ABSORB_MAX_AREA):
        continue
    pts = np.column_stack([cxs[clab == c] * CELL, cys[clab == c] * CELL])
    d, idx = tree.query(pts, k=1)
    j = int(np.argmin(d))
    if d[j] <= ABSORB_DIST + CELL:
        targ = int(big_l[idx[j]])
        if abs(z50[c] - z50[targ]) <= ABSORB_MAX_DZ:
            remap[c] = targ
            absorb_log.append((c, targ, round(float(d[j]), 2),
                               round(float(abs(z50[c] - z50[targ])), 2)))
print("absorbed:", absorb_log)
lab_p = remap[lab_p]
labels2b = remap[labels2]
area2 = np.bincount(labels2b[labels2b > 0].ravel(),
                    minlength=nlab + 1).astype(float) * CELL * CELL

# ============ 站 9：丢渣 + 发号 + 孤儿回挂 ============
keep_ids = np.nonzero(area2 >= NOISE_AREA)[0]
keep_ids = keep_ids[keep_ids > 0]
keep_ids = keep_ids[np.argsort(-area2[keep_ids])]
final_id = np.zeros(nlab + 1, np.uint32)
final_id[keep_ids] = np.arange(1, len(keep_ids) + 1, dtype=np.uint32)
inst = final_id[lab_p]
final_grid = final_id[labels2b]
dropped = [c for c in range(1, nlab + 1)
           if remap[c] == c and 0 < area2[c] < NOISE_AREA]
print("kept:", len(keep_ids), "dropped as noise:", dropped,
      [round(area2[c], 1) for c in dropped])

orphan = inst == 0
dist, (ny_i, nx_i) = ndimage.distance_transform_edt(
    final_grid == 0, return_indices=True, sampling=CELL)
oy2, ox2 = iy[orphan], ix[orphan]
od = dist[oy2, ox2]
olab = final_grid[ny_i[oy2, ox2], nx_i[oy2, ox2]]
olab[od > 1.5] = 0
n_orphan_before = int(orphan.sum())
inst[orphan] = olab
print("orphan pts:", n_orphan_before, "re-attached:",
      int((olab > 0).sum()), "left unowned:", int((inst == 0).sum()))
for k in range(1, len(keep_ids) + 1):
    print(f"  final instance {k}: {int((inst == k).sum())} pts, "
          f"area {area2[keep_ids[k-1]]:.0f} m2, z50 {np.median(z[inst == k]):.2f}")

# F7：吸收 + 终局
fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.1))
axes[0].imshow(lut2[np.clip(labels2, 0, 9)], origin="lower", extent=ext,
               interpolation="nearest")
axes[0].set_title("After splitting, before absorption: porch / junk still separate pieces")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
for (c, targ, dd, dz) in absorb_log:
    ys_, xs_ = np.nonzero(labels2 == c)
    cy_, cx_ = ys_.mean() * CELL, xs_.mean() * CELL
    yt_, xt_ = np.nonzero(labels2 == targ)
    axes[0].annotate("", xy=(xt_.mean() * CELL, yt_.mean() * CELL),
                     xytext=(cx_, cy_),
                     arrowprops=dict(arrowstyle="-|>", color="#cc3311", lw=2))
    axes[0].text(cx_ - 1.3, cy_ + 1.9,
                 f"absorbed: {dd} m away, $\\Delta$z {dz} m", fontsize=9, color="#cc3311")
img2 = np.zeros((*occ.shape, 3)); img2[...] = 1.0
fc = [(0.27, 0.47, 0.67), (0.4, 0.8, 0.87), (0.8, 0.73, 0.27), (0.13, 0.53, 0.2)]
for k in range(1, len(keep_ids) + 1):
    img2[final_grid == k] = fc[(k - 1) % 4]
axes[1].imshow(img2, origin="lower", extent=ext, interpolation="nearest")
oxp = x[src == "orphan"]; oyp = y[src == "orphan"]
axes[1].scatter(oxp, oyp, s=14, facecolors="none", edgecolors="#ee7733",
                lw=1.2, label="orphan points (re-attached to #1)")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].set_title(f"Final: {len(keep_ids)} instances (numbered by area)\n"
                  "porch merged into #1; junk cluster < 20 m$^2$ dropped; orphans re-attached")
axes[1].set_xlabel("x (m)")
for k in range(1, len(keep_ids) + 1):
    ys_, xs_ = np.nonzero(final_grid == k)
    axes[1].text(xs_.mean() * CELL, ys_.mean() * CELL, str(k),
                 ha="center", va="center", fontsize=15, fontweight="bold",
                 color="white")
fig.tight_layout(); fig.savefig(FIG + "toy_f7_final.png", dpi=150)
plt.close(fig)

# ============ 站 10（Gate A）：DTM + 离地高度 ============
DTM_CELL = 1.0
nxd = int(25 / DTM_CELL) + 2; nyd = int(18 / DTM_CELL) + 2
dtm = np.full((nyd, nxd), np.inf, np.float32)
gix = (gx / DTM_CELL).astype(np.int32); giy = (gy / DTM_CELL).astype(np.int32)
np.minimum.at(dtm, (giy, gix), gz.astype(np.float32))
valid = np.isfinite(dtm)
dtm_med = ndimage.median_filter(np.where(valid, dtm, 1e6), size=3)
dtm[valid] = np.where(np.abs(dtm[valid] - dtm_med[valid]) > 2.0,
                      dtm_med[valid], dtm[valid])
_, (iyn, ixn) = ndimage.distance_transform_edt(~valid, return_indices=True)
dtm_filled = dtm[iyn, ixn]

RING_R = 4.0
r_cells = int(RING_R / CELL)
yy, xx = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
ring_se = (xx * xx + yy * yy) <= r_cells * r_cells
hags = {}
for k in range(1, len(keep_ids) + 1):
    m = final_grid == k
    ring = ndimage.binary_dilation(m, structure=ring_se) & ~m
    ry, rx = np.nonzero(ring)
    dy = np.clip((ry * CELL / DTM_CELL).astype(np.int32), 0, nyd - 1)
    dx = np.clip((rx * CELL / DTM_CELL).astype(np.int32), 0, nxd - 1)
    vm = valid[dy, dx]
    srcv = dtm[dy[vm], dx[vm]] if vm.sum() >= 20 else dtm_filled[dy, dx]
    ground = float(np.percentile(srcv, 5))
    z50k = float(np.median(z[inst == k]))
    hags[k] = (z50k, ground, z50k - ground)
    print(f"  instance {k}: z50 {z50k:.2f}, ground P5 {ground:.2f}, "
          f"HAG {z50k - ground:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2),
                         gridspec_kw={"width_ratios": [1.15, 1]})
dtm_show = np.where(valid & (dtm < 100), dtm, np.nan)
im = axes[0].imshow(dtm_show, origin="lower",
                    extent=[0, nxd * DTM_CELL, 0, nyd * DTM_CELL],
                    cmap="copper", interpolation="nearest",
                    vmin=0, vmax=0.6)
plt.colorbar(im, ax=axes[0], shrink=0.85, label="ground elevation (m)")
axes[0].set_title("1 m DTM built from class-2 ground points\n(gaps under buildings filled from nearest neighbor)")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
ks = sorted(hags)
vals = [hags[k][2] for k in ks]
bars = axes[1].bar([f"#{k}" for k in ks], vals,
                   color=["#4477aa" if v >= 2 else "#cc3311" for v in vals])
axes[1].axhline(2.0, color="#cc3311", ls="--")
axes[1].text(3.42, 2.12, "HAG = 2 m threshold", color="#cc3311", fontsize=10, ha="right")
for b, v in zip(bars, vals):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}",
                 ha="center", fontsize=10)
axes[1].set_ylabel("height above ground, HAG (m)")
axes[1].set_title(f"Gate A: #{[k for k in ks if hags[k][2]<2][0]} (mislabeled parking lot) has HAG < 2 m, removed\nthe other three pass: 3 clean buildings")
fig.tight_layout(); fig.savefig(FIG + "toy_f8_gatea.png", dpi=150)
plt.close(fig)

stats = dict(pts=len(x), occupied=int(occ.sum()),
             speckle=int(((cnt > 0) & ~occ).sum()), raw_components=nc0,
             splits=split_info, labels_after=nlab,
             absorbed=absorb_log, dropped=[(c, round(area2[c], 1)) for c in dropped],
             final=len(keep_ids), orphan_attached=int((olab > 0).sum()),
             orphan_total=n_orphan_before,
             hags={k: [round(v, 2) for v in hags[k]] for k in hags})
json.dump(stats, open(FIG + "toy_stats.json", "w"), ensure_ascii=False, indent=1)
print("ALL FIGURES DONE")
