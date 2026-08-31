#!/usr/bin/env python
"""Cue layer — full scale (stage 5b).

Per building, from v4 plane labels + fitted plane parameters:
  RIDGE/VALLEY  3D-adjacent roof-plane pairs with normal angle > ANG_RIDGE:
                analytic intersection line clipped to boundary extent;
                classified ridge (line above face centroids) or valley (below).
                Near-parallel collinear segments are bundle-merged
                (direction < 10 deg, line distance < 0.8 m, gap < 1 m) to
                remove duplicates from ridge-cap sliver faces.
  STEP          2D detection (fixes the pilot's zero-step issue: wall points
                sit between offset roofs, so 3D kNN never links them):
                pairs of roof planes whose 0.5 m cell sets touch after
                1-cell dilation, with |dz| of the two plane models >= STEP_DZ
                at the contact cells -> line fit along the contact.
  OUTLINE       marching-squares boundary of the roof mask, simplified
                (regularization is stage 6). Stored as WKT in the original
                ftUS CRS so it overlays the LAZ in GIS.
  TOPOLOGY      edges.csv: plane adjacency with relation type.

Outputs in output_full/: cues.csv, edges.csv, outlines.csv, qa_cues_full.png
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
FT_TO_M = 0.30480060960121924

K = 12
ANG_RIDGE = 12.0
STEP_DZ = 0.25
MIN_BND = 10          # 3D boundary midpoints needed for a ridge
MIN_STEP_CELLS = 6
MIN_LEN = 1.0
BUNDLE_ANG = 10.0     # deg
BUNDLE_DIST = 0.8     # m, perpendicular line distance
BUNDLE_GAP = 1.0      # m, max gap along direction
CELL = 0.5

PILOT_IDS = [1206, 2177, 347, 264, 956, 1335, 307, 141, 34]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def plane_z(n, c, x, y):
    return c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2]


def bundle_merge(segs):
    """segs: list of dict(e1,e2,support,pa,pb,kind). Greedy merge of
    near-collinear overlapping segments; keeps the longest's line."""
    segs = sorted(segs, key=lambda s: -np.linalg.norm(s["e2"] - s["e1"]))
    out = []
    used = [False] * len(segs)
    cosb = np.cos(np.radians(BUNDLE_ANG))
    for i, s in enumerate(segs):
        if used[i]:
            continue
        d = s["e2"] - s["e1"]
        L = np.linalg.norm(d)
        if L < 1e-6:
            used[i] = True
            continue
        u = d / L
        o = s["e1"]
        tlo, thi = 0.0, L
        members = [s]
        used[i] = True
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            t = segs[j]
            dj = t["e2"] - t["e1"]
            Lj = np.linalg.norm(dj)
            if Lj < 1e-6:
                used[j] = True
                continue
            uj = dj / Lj
            if abs(float(np.dot(u[:2], uj[:2])) /
                   max(np.linalg.norm(u[:2]) * np.linalg.norm(uj[:2]), 1e-9)) < cosb:
                continue
            # perpendicular distance of t's midpoint to s's line (2D)
            mid = (t["e1"] + t["e2"]) / 2
            v = mid[:2] - o[:2]
            perp = abs(float(v[0] * u[1] - v[1] * u[0]))
            if perp > BUNDLE_DIST:
                continue
            ta = float(np.dot(t["e1"][:2] - o[:2], u[:2]))
            tb = float(np.dot(t["e2"][:2] - o[:2], u[:2]))
            lo, hi = min(ta, tb), max(ta, tb)
            if lo > thi + BUNDLE_GAP or hi < tlo - BUNDLE_GAP:
                continue
            tlo, thi = min(tlo, lo), max(thi, hi)
            members.append(t)
            used[j] = True
        e1 = o + tlo * u
        e2 = o + thi * u
        out.append(dict(e1=e1, e2=e2,
                        support=sum(m["support"] for m in members),
                        pa=s["pa"], pb=s["pb"], kind=s["kind"],
                        merged=len(members)))
    return out


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
                npts=int(r["n_points"]))

    order = np.argsort(inst, kind="stable")
    uniq, starts = np.unique(inst[order], return_index=True)
    ends = np.append(starts[1:], len(order))

    cue_rows, edge_rows, outline_rows = [], [], []
    per_building = {}
    render_cache = {}
    t0 = time.time()
    done = 0
    for fid, a, b in zip(uniq, starts, ends):
        if fid == 0:
            continue
        idx = order[a:b]
        P = X[idx]
        gp = pid[idx]
        st = styp[idx]
        roof_mask = (st == 1) & (gp > 0)
        roof_ids = np.unique(gp[roof_mask])
        ridges, steps = [], []
        if len(P) >= K + 2 and len(roof_ids) >= 1:
            # ---- 3D adjacency -> ridge/valley candidates ----
            if len(roof_ids) >= 2:
                tree = cKDTree(P)
                _, nbrs = tree.query(P, k=K + 1)
                nbrs = nbrs[:, 1:]
                pair_mids = {}
                lab = np.where(roof_mask, gp, 0)
                for i in np.nonzero(roof_mask)[0]:
                    li = lab[i]
                    for j in nbrs[i]:
                        lj = lab[j]
                        if lj == 0 or lj == li:
                            continue
                        key = (min(li, lj), max(li, lj))
                        pair_mids.setdefault(key, []).append((P[i] + P[j]) / 2)
                ridge_pairs = set()
                for (pa, pb), mids in pair_mids.items():
                    if len(mids) < MIN_BND or pa not in pl or pb not in pl:
                        continue
                    n1, c1 = pl[pa]["n"], pl[pa]["c"]
                    n2, c2 = pl[pb]["n"], pl[pb]["c"]
                    ang = np.degrees(np.arccos(min(abs(float(np.dot(n1, n2))), 1.0)))
                    if ang <= ANG_RIDGE:
                        continue
                    mids = np.array(mids)
                    bc = mids.mean(axis=0)
                    d = np.cross(n1, n2)
                    nd = np.linalg.norm(d)
                    if nd < 1e-9:
                        continue
                    d /= nd
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
                    zmid = (e1[2] + e2[2]) / 2
                    kind = "ridge" if zmid > (c1[2] + c2[2]) / 2 else "valley"
                    ridges.append(dict(e1=e1, e2=e2, support=len(mids),
                                       pa=int(pa), pb=int(pb), kind=kind))
                    ridge_pairs.add((pa, pb))
                ridges = bundle_merge(ridges)
            else:
                ridge_pairs = set()

            # ---- 2D contact -> step candidates ----
            if len(roof_ids) >= 2:
                x0m, y0m = P[:, 0].min(), P[:, 1].min()
                cells = {}
                for g in roof_ids:
                    s = roof_mask & (gp == g)
                    cx = ((P[s, 0] - x0m) / CELL).astype(np.int32)
                    cy = ((P[s, 1] - y0m) / CELL).astype(np.int32)
                    cells[g] = set(zip(cx.tolist(), cy.tolist()))
                offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                rids = sorted(cells)
                for ii in range(len(rids)):
                    for jj in range(ii + 1, len(rids)):
                        pa, pb = rids[ii], rids[jj]
                        if (pa, pb) in ridge_pairs or pa not in pl or pb not in pl:
                            continue
                        ca, cb = cells[pa], cells[pb]
                        dil_a = {(x + dx, y + dy) for x, y in ca for dx, dy in offs} \
                            if len(ca) < 20000 else ca
                        contact = dil_a & cb
                        if len(contact) < MIN_STEP_CELLS:
                            continue
                        pts2 = np.array([(x * CELL + x0m, y * CELL + y0m)
                                         for x, y in contact])
                        n1, c1 = pl[pa]["n"], pl[pa]["c"]
                        n2, c2 = pl[pb]["n"], pl[pb]["c"]
                        za = plane_z(n1, c1, pts2[:, 0], pts2[:, 1])
                        zb = plane_z(n2, c2, pts2[:, 0], pts2[:, 1])
                        dz = np.median(za - zb)
                        if abs(dz) < STEP_DZ:
                            continue
                        bc2 = pts2.mean(axis=0)
                        xy = pts2 - bc2
                        w, v = np.linalg.eigh(xy.T @ xy / len(xy))
                        u = v[:, 1]
                        t = xy @ u
                        t1, t2 = np.percentile(t, 2), np.percentile(t, 98)
                        if t2 - t1 < MIN_LEN:
                            continue
                        ztop = max(plane_z(n1, c1, *bc2), plane_z(n2, c2, *bc2))
                        e1 = np.array([bc2[0] + t1 * u[0], bc2[1] + t1 * u[1], ztop])
                        e2 = np.array([bc2[0] + t2 * u[0], bc2[1] + t2 * u[1], ztop])
                        steps.append(dict(e1=e1, e2=e2, support=len(contact),
                                          pa=int(pa), pb=int(pb),
                                          kind="step", dz=float(abs(dz))))

            # ---- outline ----
            if roof_mask.sum() >= 12:
                gx = ((P[roof_mask, 0] - P[:, 0].min()) / CELL).astype(np.int32)
                gy = ((P[roof_mask, 1] - P[:, 1].min()) / CELL).astype(np.int32)
                m = np.zeros((gy.max() + 3, gx.max() + 3), bool)
                m[gy + 1, gx + 1] = True
                m = ndimage.binary_fill_holes(
                    ndimage.binary_closing(m, structure=np.ones((3, 3), bool)))
                conts = measure.find_contours(m.astype(float), 0.5)
                if conts:
                    cont = max(conts, key=len)
                    poly = measure.approximate_polygon(cont, tolerance=1.2)
                    xy = (poly[:, ::-1] - 1) * CELL
                    xy[:, 0] += P[:, 0].min()
                    xy[:, 1] += P[:, 1].min()
                    wkt = "POLYGON((" + ",".join(
                        f"{x/FT_TO_M:.2f} {y/FT_TO_M:.2f}" for x, y in xy) + "))"
                    outline_rows.append((int(fid), len(xy), wkt))

        for s in ridges:
            cue_rows.append((int(fid), s["kind"], s["pa"], s["pb"],
                             *np.round(s["e1"], 2), *np.round(s["e2"], 2),
                             round(float(np.linalg.norm(s["e2"] - s["e1"])), 2),
                             s.get("merged", 1), ""))
            edge_rows.append((int(fid), s["pa"], s["pb"], s["kind"],
                              round(float(np.linalg.norm(s["e2"] - s["e1"])), 2), ""))
        for s in steps:
            cue_rows.append((int(fid), "step", s["pa"], s["pb"],
                             *np.round(s["e1"], 2), *np.round(s["e2"], 2),
                             round(float(np.linalg.norm(s["e2"] - s["e1"])), 2),
                             1, round(s["dz"], 2)))
            edge_rows.append((int(fid), s["pa"], s["pb"], "step",
                              round(float(np.linalg.norm(s["e2"] - s["e1"])), 2),
                              round(s["dz"], 2)))
        per_building[int(fid)] = (len(ridges), len(steps))
        if int(fid) in PILOT_IDS or len(steps) > 0:
            render_cache[int(fid)] = (P.copy(), gp.copy(), st.copy(),
                                      [dict(x) for x in ridges],
                                      [dict(x) for x in steps])
        done += 1
        if done % 400 == 0:
            log(f"{done} buildings, {time.time()-t0:.0f}s")

    log(f"done: {done} buildings in {time.time()-t0:.0f}s")
    nr = sum(1 for r in cue_rows if r[1] == "ridge")
    nv = sum(1 for r in cue_rows if r[1] == "valley")
    ns = sum(1 for r in cue_rows if r[1] == "step")
    nb_step = sum(1 for v in per_building.values() if v[1] > 0)
    log(f"cues: {nr} ridges, {nv} valleys, {ns} steps "
        f"({nb_step} buildings with steps), outlines {len(outline_rows)}")

    with open(FULL / "cues.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "type", "plane_a", "plane_b",
                    "x1", "y1", "z1", "x2", "y2", "z2", "length_m",
                    "merged_from", "dz_m"])
        w.writerows(cue_rows)
    with open(FULL / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "plane_a", "plane_b", "relation",
                    "length_m", "dz_m"])
        w.writerows(edge_rows)
    with open(FULL / "outlines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "n_vertices", "wkt_ftus"])
        w.writerows(outline_rows)

    # ---------------------------------------------------------- QA figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_ids = sorted((f for f, v in per_building.items() if v[1] > 0),
                      key=lambda f: -per_building[f][1])[:6]
    show = PILOT_IDS[:6] + step_ids
    ncol = 6
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes)
    for k, fid in enumerate(show):
        ax = axes[k // ncol, k % ncol]
        if fid not in render_cache:
            ax.axis("off")
            continue
        P, gp, st, ridges, steps = render_cache[fid]
        x0m, y0m = P[:, 0].min(), P[:, 1].min()
        Q = P - [x0m, y0m, 0]
        for g in np.unique(gp[(gp > 0) & (st == 1)]):
            s = gp == g
            ax.scatter(Q[s, 0], Q[s, 1], s=0.4, alpha=0.25, rasterized=True)
        for s in ridges:
            col = "r" if s["kind"] == "ridge" else "m"
            ax.plot([s["e1"][0] - x0m, s["e2"][0] - x0m],
                    [s["e1"][1] - y0m, s["e2"][1] - y0m], col, lw=2.2)
        for s in steps:
            ax.plot([s["e1"][0] - x0m, s["e2"][0] - x0m],
                    [s["e1"][1] - y0m, s["e2"][1] - y0m], "b", lw=2.4)
        r, sN = per_building[fid]
        ax.set_title(f"id {fid}: {r} ridge/valley, {sN} step", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Cue layer full: ridge (red) / valley (magenta) / step (blue)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FULL / "qa_cues_full.png", dpi=135)
    log("wrote qa_cues_full.png")


if __name__ == "__main__":
    main()
