#!/usr/bin/env python
"""Pilot: roof-plane extraction by region growing (Rabbani et al. 2006 style,
parameters scaled to ~30 pts/m2 per 3D BAG convention) on a stratified sample
of buildings from the cleaned instance LAZ.

Purpose: empirically verify the method choice on THIS dataset before rolling
out to all 2,283 buildings. Checks per building:
  - plausible plane count (gable=2, hip=4, complex 5-15, flat 1-3)
  - assigned-point ratio (Gate B: trees/clutter must fall out as unassigned)
  - RMS plane residual (sensor noise level ~ few cm expected)
  - wall/facade planes separated by normal direction (|nz| < 0.34)
"""

from pathlib import Path
import csv
import time
import numpy as np
import laspy
from scipy.spatial import cKDTree

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
CLEAN = BASE / "02_03_instance_seg/output_v2/buildings_clean.laz"
CSV_A = BASE / "02_03_instance_seg/output_v2/instances_gateA.csv"
OUT = BASE / "04_05_roof_planes/output_pilot"

FT_TO_M = 0.30480060960121924
K = 20                 # k-NN for normals and growing
ANG_GROW = 15.0        # deg, normal angle threshold during growth
DIST_GROW = 0.15       # m, point-to-plane distance during growth
ANG_MERGE = 8.0        # deg, merge adjacent coplanar segments
DIST_MERGE = 0.10      # m
MIN_PTS = 55           # min support (3D BAG 15 pts @ 8/m2 -> ~55 @ 30/m2)
CURV_SEED = 0.08       # max curvature for a seed
WALL_NZ = 0.34         # planes with |nz| below this are walls (slope > 70 deg)

EXPLICIT = [13, 10, 1]     # big gable complex, tall slab (facade+trees), flat industrial
STRATA = [(30, 60, 1), (80, 150, 3), (150, 250, 3), (250, 400, 2),
          (400, 1000, 2), (1000, 3000, 1)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def normals_curvature(P, nbrs):
    nb = P[nbrs]                                  # (n,k,3)
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / nb.shape[1]
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]
    flip = nrm[:, 2] < 0
    nrm[flip] *= -1
    curv = w[:, 0] / np.maximum(w.sum(axis=1), 1e-12)
    return nrm, curv


def fit_plane(pts):
    c = pts.mean(axis=0)
    q = pts - c
    cov = q.T @ q / len(pts)
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    if n[2] < 0:
        n = -n
    rms = float(np.sqrt(max(w[0], 0.0)))
    return n, c, rms


def segment(P):
    n = len(P)
    tree = cKDTree(P)
    _, nbrs = tree.query(P, k=K + 1)
    nbrs = nbrs[:, 1:]
    nrm, curv = normals_curvature(P, nbrs)
    cos_grow = np.cos(np.radians(ANG_GROW))
    labels = np.full(n, -1, np.int32)
    next_lab = 0
    order = np.argsort(curv)
    for seed in order:
        if labels[seed] != -1 or curv[seed] > CURV_SEED:
            continue
        rn = nrm[seed].copy(); rc = P[seed].copy()
        region = [seed]
        labels[seed] = -2                      # provisional
        queue = [seed]
        since_fit = 0
        while queue:
            i = queue.pop()
            for j in nbrs[i]:
                if labels[j] != -1:
                    continue
                if abs(np.dot(nrm[j], rn)) < cos_grow:
                    continue
                if abs(np.dot(P[j] - rc, rn)) > DIST_GROW:
                    continue
                labels[j] = -2
                region.append(j)
                queue.append(j)
                since_fit += 1
                if since_fit >= 256:
                    rn, rc, _ = fit_plane(P[region])
                    since_fit = 0
        if len(region) >= MIN_PTS:
            labels[np.array(region)] = next_lab
            next_lab += 1
        else:
            labels[np.array(region)] = -1
    # plane params
    planes = {}
    for lab in range(next_lab):
        pts = P[labels == lab]
        nvec, c, rms = fit_plane(pts)
        planes[lab] = [nvec, c, rms, len(pts)]
    # merge adjacent coplanar segments
    cos_merge = np.cos(np.radians(ANG_MERGE))
    changed = True
    it = 0
    while changed and it < 3:
        changed = False; it += 1
        lab_n = labels[nbrs]                    # (n,k)
        mine = labels[:, None]
        mask = (lab_n >= 0) & (mine >= 0) & (lab_n != mine)
        pairs = set()
        a = np.broadcast_to(mine, lab_n.shape)[mask]
        b = lab_n[mask]
        for x, y in zip(a.tolist(), b.tolist()):
            pairs.add((min(x, y), max(x, y)))
        remap = {l: l for l in planes}
        for x, y in sorted(pairs):
            rx, ry = remap[x], remap[y]
            if rx == ry:
                continue
            n1, c1, _, k1 = planes[rx]
            n2, c2, _, k2 = planes[ry]
            if abs(np.dot(n1, n2)) < cos_merge:
                continue
            if abs(np.dot(c2 - c1, n1)) > DIST_MERGE or abs(np.dot(c1 - c2, n2)) > DIST_MERGE:
                continue
            tgt, src = (rx, ry) if k1 >= k2 else (ry, rx)
            labels[labels == src] = tgt
            pts = P[labels == tgt]
            planes[tgt] = [*fit_plane(pts), len(pts)]
            del planes[src]
            for kk, vv in remap.items():
                if vv == src:
                    remap[kk] = tgt
            changed = True
    # one reassignment sweep for leftovers
    cos_assign = np.cos(np.radians(25.0))
    un = np.nonzero(labels < 0)[0]
    for i in un:
        for j in nbrs[i]:
            lab = labels[j]
            if lab < 0:
                continue
            nvec, c, _, _ = planes[lab]
            if abs(np.dot(P[i] - c, nvec)) <= DIST_GROW and abs(np.dot(nrm[i], nvec)) >= cos_assign:
                labels[i] = lab
                break
    # refresh counts
    for lab in list(planes):
        k = int((labels == lab).sum())
        if k == 0:
            del planes[lab]
        else:
            planes[lab][3] = k
    return labels, planes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CSV_A) as f:
        rows = [r for r in csv.DictReader(f) if r["decision"].startswith("keep")]
    rng = np.random.default_rng(0)
    chosen = list(EXPLICIT)
    for lo, hi, k in STRATA:
        pool = [int(r["instance_id"]) for r in rows
                if lo <= float(r["area_m2"]) < hi and int(r["instance_id"]) not in chosen]
        chosen += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    log(f"sample: {chosen}")

    las = laspy.read(CLEAN)
    inst = np.asarray(las.instance_id)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                         np.asarray(las.z)]) * FT_TO_M

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    ncol = 5
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
    axes = np.atleast_2d(axes)
    stats = []
    for k, fid in enumerate(chosen):
        m = inst == fid
        P = X[m]
        t0 = time.time()
        labels, planes = segment(P)
        dt = time.time() - t0
        roof = {l: p for l, p in planes.items() if abs(p[0][2]) >= WALL_NZ}
        wall = {l: p for l, p in planes.items() if abs(p[0][2]) < WALL_NZ}
        assigned = float((labels >= 0).mean())
        roof_pts = int(sum(p[3] for p in roof.values()))
        rms_mean = float(np.mean([p[2] for p in roof.values()])) if roof else 0.0
        stats.append((fid, len(P), len(roof), len(wall), assigned,
                      roof_pts / len(P), rms_mean, dt))
        ax = axes[k // ncol, k % ncol]
        un = labels < 0
        ax.scatter(P[un, 0] - P[:, 0].min(), P[un, 1] - P[:, 1].min(),
                   s=0.5, c="0.82", rasterized=True)
        cmap = cm.get_cmap("tab20")
        for i, lab in enumerate(sorted(planes)):
            sel = labels == lab
            col = "k" if lab in wall else cmap(i % 20)
            ax.scatter(P[sel, 0] - P[:, 0].min(), P[sel, 1] - P[:, 1].min(),
                       s=0.6, color=col, rasterized=True)
        ax.set_title(f"id {fid} | {len(P):,} pts | {len(roof)} roof + "
                     f"{len(wall)} wall planes\nassigned {assigned*100:.0f}% | "
                     f"RMS {rms_mean*100:.1f} cm | {dt:.1f}s", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        log(f"id {fid}: {len(P):,} pts, {len(roof)} roof / {len(wall)} wall, "
            f"assigned {assigned*100:.0f}%, rms {rms_mean*100:.1f} cm, {dt:.1f}s")
    for k in range(len(chosen), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "pilot_planes.png", dpi=140)
    with open(OUT / "pilot_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "n_points", "n_roof_planes", "n_wall_planes",
                    "assigned_ratio", "roof_pt_ratio", "rms_m", "seconds"])
        w.writerows(stats)
    log("done")


if __name__ == "__main__":
    main()
