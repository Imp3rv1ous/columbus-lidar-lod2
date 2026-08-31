#!/usr/bin/env python
"""Cue layer pilot: per-building roof structure cues from the full-run planes.

For each selected building, derive from plane labels + fitted plane params:
  RIDGE / VALLEY  adjacent roof-plane pairs with normal angle > ANG_RIDGE:
                  analytic plane-plane intersection line, clipped to the
                  extent of the pair's boundary points
  STEP            near-parallel adjacent pairs with vertical offset >= STEP_DZ
                  at their common boundary: 2D line fit along the boundary
  OUTLINE         boundary of the rasterized roof mask (marching squares);
                  regularization comes later (stage 6)
  TOPOLOGY        graph: nodes = roof planes, edges = ridge/step relations

Outputs: cues CSV + a 3x3 visual (points faint, ridges red, steps blue,
outline black) for the sample buildings.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
FULL = BASE / "04_05_roof_planes/output_full"
OUT = FULL
FT_TO_M = 0.30480060960121924

SAMPLE = [1206, 2177, 347, 264, 956, 1335, 307, 141, 34]
K = 12
ANG_RIDGE = 12.0     # deg: pairs above this are intersection (ridge/valley)
STEP_DZ = 0.25       # m: parallel pairs offset more than this are steps
MIN_BND_PTS = 10
MIN_LEN = 1.0        # m


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    las = laspy.read(FULL / "buildings_planes.laz")
    inst = np.asarray(las.instance_id)
    pid = np.asarray(las.plane_id)
    styp = np.asarray(las.surf_type)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                         np.asarray(las.z)]) * FT_TO_M

    pl = {}
    with open(FULL / "planes.csv") as f:
        for r in csv.DictReader(f):
            pl[int(r["plane_id"])] = dict(
                inst=int(r["instance_id"]), typ=r["type"],
                n=np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])]),
                c=np.array([float(r["cx_m"]), float(r["cy_m"]), float(r["cz_m"])]),
                slope=float(r["slope_deg"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    cue_rows = []
    for k, fid in enumerate(SAMPLE):
        ax = axes[k // 3, k % 3]
        sel = inst == fid
        P = X[sel]
        gp = pid[sel]
        st = styp[sel]
        x0, y0 = P[:, 0].min(), P[:, 1].min()
        Q = P - [x0, y0, 0]

        roof_ids = [g for g in np.unique(gp[(gp > 0) & (st == 1)])]
        tree = cKDTree(P)
        _, nbrs = tree.query(P, k=K + 1)
        nbrs = nbrs[:, 1:]

        # boundary midpoints per adjacent roof-plane pair
        pairs = {}
        lab = gp
        for i in range(len(P)):
            if st[i] != 1 or lab[i] == 0:
                continue
            for j in nbrs[i]:
                if st[j] != 1 or lab[j] == 0 or lab[j] == lab[i]:
                    continue
                key = (min(lab[i], lab[j]), max(lab[i], lab[j]))
                pairs.setdefault(key, []).append((P[i] + P[j]) / 2)

        ridges, steps = [], []
        for (a, b), mids in pairs.items():
            if len(mids) < MIN_BND_PTS or a not in pl or b not in pl:
                continue
            mids = np.array(mids)
            bc = mids.mean(axis=0)
            n1, c1 = pl[a]["n"], pl[a]["c"]
            n2, c2 = pl[b]["n"], pl[b]["c"]
            ang = np.degrees(np.arccos(min(abs(float(np.dot(n1, n2))), 1.0)))
            if ang > ANG_RIDGE:
                d = np.cross(n1, n2)
                d /= np.linalg.norm(d)
                A = np.vstack([n1, n2, d])
                rhs = np.array([np.dot(n1, c1), np.dot(n2, c2), np.dot(d, bc)])
                try:
                    x0l = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    continue
                t = (mids - x0l) @ d
                t1, t2 = np.percentile(t, 2), np.percentile(t, 98)
                if t2 - t1 < MIN_LEN:
                    continue
                e1, e2 = x0l + t1 * d, x0l + t2 * d
                ridges.append((a, b, e1, e2))
            else:
                # vertical offset of plane b vs plane a at the boundary
                z1 = c1[2] - (n1[0] * (bc[0] - c1[0]) + n1[1] * (bc[1] - c1[1])) / n1[2]
                z2 = c2[2] - (n2[0] * (bc[0] - c2[0]) + n2[1] * (bc[1] - c2[1])) / n2[2]
                if abs(z1 - z2) < STEP_DZ:
                    continue
                xy = mids[:, :2] - bc[:2]
                w, v = np.linalg.eigh(xy.T @ xy / len(xy))
                u = v[:, 1]
                t = xy @ u
                t1, t2 = np.percentile(t, 2), np.percentile(t, 98)
                if t2 - t1 < MIN_LEN:
                    continue
                zt = max(z1, z2)
                e1 = np.array([bc[0] + t1 * u[0], bc[1] + t1 * u[1], zt])
                e2 = np.array([bc[0] + t2 * u[0], bc[1] + t2 * u[1], zt])
                steps.append((a, b, e1, e2, abs(z1 - z2)))

        # outline from roof mask
        rm = (st == 1) & (gp > 0)
        gx = ((P[rm, 0] - x0) / 0.5).astype(np.int32)
        gy = ((P[rm, 1] - y0) / 0.5).astype(np.int32)
        m = np.zeros((gy.max() + 3, gx.max() + 3), bool)
        m[gy + 1, gx + 1] = True
        m = ndimage.binary_fill_holes(
            ndimage.binary_closing(m, structure=np.ones((3, 3), bool)))
        outline = []
        for cont in measure.find_contours(m.astype(float), 0.5):
            if len(cont) < 8:
                continue
            poly = measure.approximate_polygon(cont, tolerance=1.2)
            outline.append((poly[:, ::-1] - 1) * 0.5)  # (col,row)->x,y m
        outline.sort(key=lambda p: -len(p))

        # plot
        for g in roof_ids:
            s = gp == g
            ax.scatter(Q[s, 0], Q[s, 1], s=0.4, alpha=0.25, rasterized=True)
        unl = (gp == 0) | (st == 2)
        ax.scatter(Q[unl, 0], Q[unl, 1], s=0.3, c="0.9", rasterized=True)
        for o in outline[:3]:
            ax.plot(o[:, 0], o[:, 1], "k-", lw=1.4)
        for a, b, e1, e2 in ridges:
            ax.plot([e1[0] - x0, e2[0] - x0], [e1[1] - y0, e2[1] - y0],
                    "r-", lw=2.4)
            cue_rows.append((fid, "ridge", a, b, *np.round(e1, 2), *np.round(e2, 2),
                             round(float(np.linalg.norm(e2 - e1)), 2)))
        for a, b, e1, e2, dz in steps:
            ax.plot([e1[0] - x0, e2[0] - x0], [e1[1] - y0, e2[1] - y0],
                    "b-", lw=2.4)
            cue_rows.append((fid, f"step(dz={dz:.2f})", a, b,
                             *np.round(e1, 2), *np.round(e2, 2),
                             round(float(np.linalg.norm(e2 - e1)), 2)))
        # topology graph inset: plane centroids + edges
        for g in roof_ids:
            c = pl[g]["c"]
            ax.plot(c[0] - x0, c[1] - y0, "o", ms=5, mfc="yellow", mec="k")
        for a, b, e1, e2 in ridges:
            ca, cb = pl[a]["c"], pl[b]["c"]
            ax.plot([ca[0] - x0, cb[0] - x0], [ca[1] - y0, cb[1] - y0],
                    "r--", lw=0.8, alpha=0.7)
        for a, b, e1, e2, dz in steps:
            ca, cb = pl[a]["c"], pl[b]["c"]
            ax.plot([ca[0] - x0, cb[0] - x0], [ca[1] - y0, cb[1] - y0],
                    "b--", lw=0.8, alpha=0.7)
        ax.set_title(f"id {fid}: {len(ridges)} ridge, {len(steps)} step, "
                     f"{len(roof_ids)} faces", fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        log(f"id {fid}: faces {len(roof_ids)}, ridges {len(ridges)}, "
            f"steps {len(steps)}, outline pts {len(outline[0]) if outline else 0}")

    fig.suptitle("Cue layer: ridges (red), steps (blue), outline (black), "
                 "topology (dashed, nodes=plane centroids)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "qa_cue_layer.png", dpi=140)
    with open(OUT / "cues_pilot.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "type", "plane_a", "plane_b",
                    "x1", "y1", "z1", "x2", "y2", "z2", "length_m"])
        w.writerows(cue_rows)
    log("done")


if __name__ == "__main__":
    main()
