#!/usr/bin/env python
"""Ablation: region growing (Rabbani/3D BAG style) vs Jung/Jwa/Sohn 2017
(height clustering delta_h=1 m -> sequential RANSAC zeta=0.1 m) on the SAME
15-building stratified sample, same min support (55 pts).

Extra metric: spatial coherence of each plane's inliers — a plane whose
inlier set splits into >=2 substantial disconnected chunks (0.5 m grid,
8-conn) is a "phantom" plane (the classic sequential-RANSAC failure mode;
region growing cannot produce one by construction — verified, not assumed).
"""

from pathlib import Path
import csv
import sys
import time
import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from pilot_region_growing import segment as rg_segment, EXPLICIT, STRATA, WALL_NZ

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
CLEAN = BASE / "02_03_instance_seg/output_v2/buildings_clean.laz"
CSV_A = BASE / "02_03_instance_seg/output_v2/instances_gateA.csv"
OUT = BASE / "04_05_roof_planes/output_pilot"

FT_TO_M = 0.30480060960121924
MIN_PTS = 55
DH = 1.0        # Jung height-clustering threshold (paper: delta_h = 1 m)
ZETA = 0.10     # Jung RANSAC tolerance (paper: 0.1 m)
RANSAC_ITERS = 500
K = 20


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def height_clusters(P, nbrs, dh=DH):
    n = len(P)
    lab = np.full(n, -1, np.int32)
    order = np.argsort(-P[:, 2])
    cur = 0
    for seed in order:
        if lab[seed] != -1:
            continue
        lab[seed] = cur
        queue = [seed]
        while queue:
            i = queue.pop()
            zi = P[i, 2]
            for j in nbrs[i]:
                if lab[j] == -1 and abs(P[j, 2] - zi) < dh:
                    lab[j] = cur
                    queue.append(j)
        cur += 1
    return lab, cur


def seq_ransac(Q, rng, min_pts=MIN_PTS, tol=ZETA, iters=RANSAC_ITERS):
    """Sequential RANSAC on one height cluster. Returns labels (-1 unassigned)
    and list of (normal, centroid, rms, npts)."""
    n = len(Q)
    labels = np.full(n, -1, np.int32)
    remaining = np.arange(n)
    planes = []
    while len(remaining) >= min_pts:
        R = Q[remaining]
        m = len(R)
        idx = rng.integers(0, m, size=(iters, 3))
        p0, p1, p2 = R[idx[:, 0]], R[idx[:, 1]], R[idx[:, 2]]
        nv = np.cross(p1 - p0, p2 - p0)
        nrm = np.linalg.norm(nv, axis=1)
        ok = nrm > 1e-9
        nv = nv[ok] / nrm[ok, None]
        d0 = -(nv * p0[ok]).sum(1)
        best_cnt, best = 0, None
        for s in range(0, len(nv), 64):
            dist = np.abs(R @ nv[s:s + 64].T + d0[s:s + 64])
            cnt = (dist <= tol).sum(axis=0)
            j = int(np.argmax(cnt))
            if cnt[j] > best_cnt:
                best_cnt = int(cnt[j])
                best = (nv[s + j], d0[s + j])
        if best is None or best_cnt < min_pts:
            break
        inl = np.abs(R @ best[0] + best[1]) <= tol
        pts = R[inl]                      # LSQ refit, recount
        c = pts.mean(axis=0)
        q = pts - c
        w, v = np.linalg.eigh(q.T @ q / len(pts))
        nvec = v[:, 0]
        if nvec[2] < 0:
            nvec = -nvec
        inl = np.abs((R - c) @ nvec) <= tol
        if int(inl.sum()) < min_pts:
            break
        pts = R[inl]
        c = pts.mean(axis=0)
        q = pts - c
        w, v = np.linalg.eigh(q.T @ q / len(pts))
        nvec = v[:, 0]
        if nvec[2] < 0:
            nvec = -nvec
        rms = float(np.sqrt(max(w[0], 0.0)))
        labels[remaining[inl]] = len(planes)
        planes.append([nvec, c, rms, int(inl.sum())])
        remaining = remaining[~inl]
    return labels, planes


def jung_segment(P, rng):
    tree = cKDTree(P)
    _, nbrs = tree.query(P, k=K + 1)
    nbrs = nbrs[:, 1:]
    hlab, nh = height_clusters(P, nbrs)
    labels = np.full(len(P), -1, np.int32)
    planes = {}
    nxt = 0
    for h in range(nh):
        sel = np.nonzero(hlab == h)[0]
        if len(sel) < MIN_PTS:
            continue
        sub_lab, sub_planes = seq_ransac(P[sel], rng)
        for i, pl in enumerate(sub_planes):
            labels[sel[sub_lab == i]] = nxt
            planes[nxt] = pl
            nxt += 1
    return labels, planes


def incoherent_planes(P, labels, planes):
    """count planes whose inliers form >=2 substantial disconnected chunks"""
    bad = 0
    for lab in planes:
        sel = labels == lab
        if sel.sum() < MIN_PTS:
            continue
        gx = ((P[sel, 0] - P[sel, 0].min()) / 0.5).astype(np.int32)
        gy = ((P[sel, 1] - P[sel, 1].min()) / 0.5).astype(np.int32)
        m = np.zeros((gy.max() + 1, gx.max() + 1), bool)
        m[gy, gx] = True
        cc, k = ndimage.label(m, structure=np.ones((3, 3), bool))
        if k < 2:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        thr = max(20, 0.10 * sizes.sum())
        if (sizes >= thr).sum() >= 2:
            bad += 1
    return bad


def main():
    with open(CSV_A) as f:
        rows = [r for r in csv.DictReader(f) if r["decision"].startswith("keep")]
    rng0 = np.random.default_rng(0)
    chosen = list(EXPLICIT)
    for lo, hi, k in STRATA:
        pool = [int(r["instance_id"]) for r in rows
                if lo <= float(r["area_m2"]) < hi and int(r["instance_id"]) not in chosen]
        chosen += list(rng0.choice(pool, size=min(k, len(pool)), replace=False))
    log(f"sample: {chosen}")

    las = laspy.read(CLEAN)
    inst = np.asarray(las.instance_id)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                         np.asarray(las.z)]) * FT_TO_M

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    cmap = colormaps["tab20"]
    nrow, ncol = 5, 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.9 * nrow))
    stats = []
    rng = np.random.default_rng(7)
    for k, fid in enumerate(chosen):
        P = X[inst == fid]
        res = {}
        for meth, fn in (("RG", lambda p: rg_segment(p)),
                         ("Jung", lambda p: jung_segment(p, rng))):
            t0 = time.time()
            labels, planes = fn(P)
            dt = time.time() - t0
            roof = {l: p for l, p in planes.items() if abs(p[0][2]) >= WALL_NZ}
            wall = {l: p for l, p in planes.items() if abs(p[0][2]) < WALL_NZ}
            bad = incoherent_planes(P, labels, roof)
            res[meth] = dict(labels=labels, planes=planes, roof=roof, wall=wall,
                             assigned=float((labels >= 0).mean()), dt=dt,
                             rms=float(np.mean([p[2] for p in roof.values()]))
                             if roof else 0.0, bad=bad)
            stats.append((fid, meth, len(P), len(roof), len(wall),
                          round(res[meth]["assigned"], 3),
                          round(res[meth]["rms"] * 100, 1), bad, round(dt, 1)))
            log(f"id {fid} [{meth}]: {len(roof)} roof/{len(wall)} wall, "
                f"assigned {res[meth]['assigned']*100:.0f}%, "
                f"rms {res[meth]['rms']*100:.1f} cm, incoherent {bad}, {dt:.1f}s")
        for col, meth in enumerate(("RG", "Jung")):
            ax = axes[(2 * k) // ncol, (2 * k) % ncol + col] if False else \
                 axes[(2 * k + col) // ncol, (2 * k + col) % ncol]
            r = res[meth]
            labels = r["labels"]
            un = labels < 0
            ax.scatter(P[un, 0] - P[:, 0].min(), P[un, 1] - P[:, 1].min(),
                       s=0.4, c="0.85", rasterized=True)
            for i, lab in enumerate(sorted(r["planes"])):
                sel = labels == lab
                col_c = "k" if lab in r["wall"] else cmap(i % 20)
                ax.scatter(P[sel, 0] - P[:, 0].min(), P[sel, 1] - P[:, 1].min(),
                           s=0.5, color=col_c, rasterized=True)
            ax.set_title(f"id {fid} [{meth}] {len(r['roof'])}rf/{len(r['wall'])}w "
                         f"| {r['assigned']*100:.0f}% | rms {r['rms']*100:.1f}cm "
                         f"| phantom {r['bad']}", fontsize=8)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "rg_vs_jung.png", dpi=130)
    with open(OUT / "rg_vs_jung_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "method", "n_points", "n_roof", "n_wall",
                    "assigned", "rms_cm", "incoherent_planes", "seconds"])
        w.writerows(stats)
    # aggregate
    for meth in ("RG", "Jung"):
        ss = [s for s in stats if s[1] == meth]
        log(f"[{meth}] mean assigned {np.mean([s[5] for s in ss])*100:.1f}%  "
            f"mean rms {np.mean([s[6] for s in ss]):.1f} cm  "
            f"total incoherent planes {sum(s[7] for s in ss)}  "
            f"total time {sum(s[8] for s in ss):.0f}s")
    log("done")


if __name__ == "__main__":
    main()
