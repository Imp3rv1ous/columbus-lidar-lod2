#!/usr/bin/env python
"""Vegetation-evidence filter for extracted planes (Gate B made explicit).

For every plane: veg_ratio = (# class-5 vegetation points falling in the
plane's 1 m cells, dilated by 1, within |z - plane(z at cell)| <= Z_WIN)
divided by the plane's own point count. A real roof face keeps veg below
the eave outside the z-window (ratio ~ 0); a canopy patch fitted as a plane
sits INSIDE the same crown as thousands of class-5 points (ratio >> 1).

Kill rules (applied later, this script measures + renders evidence):
  plane-level: veg_ratio >= R_KILL and (npts < NPTS_SAFE or rms > RMS_SAFE
               or wall-type)   -> vegetation plane
  instance-level: after plane filtering, if (killed + unassigned) points
               >= INST_KILL_FRAC of the instance -> whole instance is
               vegetation/clutter (Gate A retro-verdict, reviewed manually)

Run with APPLY=False first: writes plane_veg.csv, instance_veg.csv and a
QA render of top candidates for manual review.
"""

import csv
import time
from pathlib import Path

import numpy as np
import laspy

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
FULL = BASE / "04_05_roof_planes/output_full"
PRED = BASE / "01_semantic_randlanet/results/preds"
FT_TO_M = 0.30480060960121924

CELL = 1.0
Z_WIN = 2.0
VEG_CLASS = 5
R_KILL = 1.0
NPTS_SAFE = 300
RMS_SAFE = 0.035
INST_KILL_FRAC = 0.90


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    # ---------------- veg cell hash (global bounds from tile headers)
    tiles = sorted(PRED.glob("*_pred.laz"))
    gmin = np.array([np.inf, np.inf])
    for t in tiles:
        with laspy.open(t) as r:
            gmin = np.minimum(gmin, r.header.mins[:2])
    x0, y0 = gmin * FT_TO_M
    vx, vy, vz = [], [], []
    for t in tiles:
        las = laspy.read(t)
        m = np.asarray(las.classification) == VEG_CLASS
        vx.append(np.asarray(las.x)[m]); vy.append(np.asarray(las.y)[m])
        vz.append(np.asarray(las.z)[m])
    vx = np.concatenate(vx) * FT_TO_M
    vy = np.concatenate(vy) * FT_TO_M
    vzm = (np.concatenate(vz) * FT_TO_M).astype(np.float32)
    log(f"veg points: {len(vx):,}")
    nx = int((vx.max() - x0) / CELL) + 2
    cid = ((vy - y0) / CELL).astype(np.int64) * nx + ((vx - x0) / CELL).astype(np.int64)
    o = np.argsort(cid, kind="stable")
    cid_s = cid[o]; vz_s = vzm[o]
    uc, st = np.unique(cid_s, return_index=True)
    en = np.append(st[1:], len(cid_s))
    cell_slice = dict(zip(uc.tolist(), zip(st.tolist(), en.tolist())))
    log(f"veg cells: {len(uc):,}")

    # ---------------- planes + building points
    las = laspy.read(FULL / "buildings_planes.laz")
    pid = np.asarray(las.plane_id)
    styp = np.asarray(las.surf_type)
    inst = np.asarray(las.instance_id)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                         np.asarray(las.z)]) * FT_TO_M
    pl = {}
    with open(FULL / "planes.csv") as f:
        for r in csv.DictReader(f):
            pl[int(r["plane_id"])] = r

    order = np.argsort(pid, kind="stable")
    u, s = np.unique(pid[order], return_index=True)
    e = np.append(s[1:], len(order))
    rows = []
    t0 = time.time()
    for g, a, b in zip(u, s, e):
        if g == 0:
            continue
        idx = order[a:b]
        P = X[idx]
        r = pl[int(g)]
        n = np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])])
        c = np.array([float(r["cx_m"]), float(r["cy_m"]), float(r["cz_m"])])
        cxs = ((P[:, 0] - x0) / CELL).astype(np.int64)
        cys = ((P[:, 1] - y0) / CELL).astype(np.int64)
        cells = set(zip(cxs.tolist(), cys.tolist()))
        dil = {(x + dx, y + dy) for x, y in cells
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        veg_near = veg_below = 0
        nz = n[2] if abs(n[2]) > 0.05 else (0.05 if n[2] >= 0 else -0.05)
        for x_c, y_c in dil:
            k = cell_slice.get(y_c * nx + x_c)
            if k is None:
                continue
            zv = vz_s[k[0]:k[1]]
            xc = x0 + (x_c + 0.5) * CELL
            yc = y0 + (y_c + 0.5) * CELL
            zp = c[2] - (n[0] * (xc - c[0]) + n[1] * (yc - c[1])) / nz
            dzv = zv - zp
            veg_near += int(np.count_nonzero(np.abs(dzv) <= Z_WIN))
            veg_below += int(np.count_nonzero((dzv <= -0.3) & (dzv >= -Z_WIN)))
        npts = len(idx)
        ratio = veg_near / npts
        below = veg_below / npts
        wall = styp[idx[0]] == 2
        rms = float(r["rms_m"])
        # canopy-interior patches have veg on BOTH sides of the plane; a real
        # roof under overhanging trees has veg only above -> require
        # below-plane evidence before killing
        kill = (ratio >= R_KILL and below >= 0.3
                and (npts < NPTS_SAFE or rms > RMS_SAFE or wall))
        rows.append((int(g), int(inst[idx[0]]), r["type"], npts,
                     round(float(r["area_m2"]), 1), rms, veg_near,
                     round(ratio, 2), round(below, 2), int(kill)))
    log(f"measured {len(rows)} planes in {time.time()-t0:.0f}s")
    with open(FULL / "plane_veg.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane_id", "instance_id", "type", "n_points", "area_m2",
                    "rms_m", "veg_near", "veg_ratio", "veg_below_ratio", "kill"])
        w.writerows(rows)

    killed = [r for r in rows if r[9]]
    log(f"plane verdicts: kill {len(killed)} / {len(rows)} "
        f"({sum(r[3] for r in killed):,} points)")

    # instance-level: killed+unassigned fraction
    per_inst_pts = {}
    per_inst_killed = {}
    for fid in np.unique(inst[inst > 0]):
        per_inst_pts[int(fid)] = int((inst == fid).sum())
    for r in rows:
        if r[9]:
            per_inst_killed[r[1]] = per_inst_killed.get(r[1], 0) + r[3]
    unass = {}
    for fid in per_inst_pts:
        m = inst == fid
        unass[fid] = int((pid[m] == 0).sum())
    inst_rows = []
    for fid, tot in per_inst_pts.items():
        bad = per_inst_killed.get(fid, 0) + unass[fid]
        frac = bad / tot
        inst_rows.append((fid, tot, per_inst_killed.get(fid, 0), unass[fid],
                          round(frac, 3), int(frac >= INST_KILL_FRAC)))
    inst_rows.sort(key=lambda r: -r[4])
    with open(FULL / "instance_veg.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "n_points", "killed_plane_pts",
                    "unassigned_pts", "bad_frac", "kill_candidate"])
        w.writerows(inst_rows)
    cand = [r for r in inst_rows if r[5]]
    log(f"instance kill candidates: {len(cand)}")
    for r in cand[:15]:
        log(f"  id {r[0]:5d}: {r[1]:6d} pts, bad_frac {r[4]}")

    # ---------------- QA render: top killed planes + instance candidates
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_killed = sorted(killed, key=lambda r: -r[3])[:9]
    show = [("plane", r[0], r[1]) for r in top_killed] + \
           [("inst", None, r[0]) for r in cand[:9]]
    ncol = 6
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.8 * nrow))
    axes = np.atleast_2d(axes)
    for k, (kind, g, fid) in enumerate(show):
        ax = axes[k // ncol, k % ncol]
        m = inst == fid
        P = X[m]
        cx, cy = np.median(P[:, 0]), np.median(P[:, 1])
        half = 30.0
        wv = (np.abs(vx - cx) < half) & (np.abs(vy - cy) < half)
        ax.scatter(vx[wv] - cx, vy[wv] - cy, s=0.3, c="#7fbf7f",
                   alpha=0.4, rasterized=True)
        ax.scatter(P[:, 0] - cx, P[:, 1] - cy, s=0.6, c="0.55", rasterized=True)
        if kind == "plane":
            sel = pid == g
            ax.scatter(X[sel, 0] - cx, X[sel, 1] - cy, s=1.2, c="red",
                       rasterized=True)
            ax.set_title(f"KILL plane {g} (inst {fid})", fontsize=8)
        else:
            ax.set_title(f"KILL-CAND instance {fid}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(show), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("Vegetation-evidence verdicts: green=class-5 veg, gray=building "
                 "pts, red=plane to kill", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FULL / "qa_veg_filter.png", dpi=135)
    log("wrote qa_veg_filter.png")

    hist = np.array([r[7] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(hist, 0, 8), bins=80, color="#4477aa")
    ax.axvline(R_KILL, color="r", ls="--", label=f"R_KILL={R_KILL}")
    ax.set_xlabel("veg_ratio"); ax.set_ylabel("planes"); ax.set_yscale("log")
    ax.legend(); fig.tight_layout()
    fig.savefig(FULL / "qa_veg_ratio_hist.png", dpi=140)
    log("done")


if __name__ == "__main__":
    main()
