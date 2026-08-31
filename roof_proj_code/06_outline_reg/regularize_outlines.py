#!/usr/bin/env python
"""Stage 6 — building outline regularization.

Per building:
  1. Roof mask (0.5 m) -> full-resolution marching-squares contour.
  2. Dominant direction theta from the building's RIDGE/VALLEY lines
     (length-weighted, folded mod 90; ridges are plane-intersection lines,
     far more accurate than the jagged boundary). Fallback: length-weighted
     mod-90 direction histogram of the contour segments.
  3. Douglas-Peucker split (0.75 m) -> each segment snapped to theta or
     theta+90 when within SNAP_ANG; persistent deviants stay free
     ("diagonal", protects 45-degree bays etc.).
  4. Consecutive same-direction near-collinear segments merged; line offset
     re-fit from the raw contour points it supports.
  5. Polygon rebuilt by intersecting consecutive lines (perpendicular
     connector inserted between near-parallel neighbors).
  6. Safety net: self-intersecting or area-change>25% results fall back to
     the unregularized contour and are flagged.

References: Sampath & Shan (2007) PE&RS 73(7) constrained-LSQ boundary
regularization; Huang et al. (2022) City3D orientation clustering +
adjustment; Douglas & Peucker (1973).

Outputs (output/): outlines_regularized.csv (WKT ftUS + quality metrics),
outlines_regularized.obj (CloudCompare overlay), qa_outline_reg.png,
qa_outline_metrics.png.
"""

import csv
import math
import time
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage
from skimage import measure

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
FULL = BASE / "04_05_roof_planes/output_full"
OUT = BASE / "06_outline_reg/output"
FT_TO_M = 0.30480060960121924

CELL = 0.5
DP_TOL = 1.5          # cells (= 0.75 m) segment split tolerance
SNAP_ANG = 15.0       # deg, snap segment to theta / theta+90 within this
COLL_OFF = 0.4        # m, collinear merge offset
MIN_SEG = 1.0         # m, shorter merged runs get absorbed
PAR_LIMIT = 20.0      # deg, consecutive lines closer than this get connector
AREA_TOL = 0.25
OVERSHOOT = 2.2       # m, max distance corner may sit from both support ends
SPIKE_ANG = 45.0      # deg, acute tips below this get removed (no exemptions)
SPIKE_EDGE = 5.0      # m, spike removal needs one adjacent edge below this
LINE_MERGE_ANG = 20.0 # deg, near-collinear consecutive lines merge...
LINE_MERGE_OFF = 0.4  # m, ...when offset below this (kills micro-jogs)       # fallback if area changes more than this fraction
PILOT = [1206, 2177, 347, 264, 956, 1335, 307, 141, 34]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fold_mean_dir(angles_deg, weights):
    """weighted mean direction folded to [0, 90)"""
    a = np.radians(np.asarray(angles_deg) * 4.0)
    w = np.asarray(weights)
    s = (w * np.sin(a)).sum(); c = (w * np.cos(a)).sum()
    return (math.degrees(math.atan2(s, c)) / 4.0) % 90.0


def ang_dist(a, b, period=180.0):
    d = abs((a - b) % period)
    return min(d, period - d)


def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def self_intersects(pts):
    n = len(pts)
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]

    def inter(p1, p2, p3, p4):
        d1 = np.cross(p4 - p3, p1 - p3); d2 = np.cross(p4 - p3, p2 - p3)
        d3 = np.cross(p2 - p1, p3 - p1); d4 = np.cross(p2 - p1, p4 - p1)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if inter(segs[i][0], segs[i][1], segs[j][0], segs[j][1]):
                return True
    return False



def corner_regularity(P):
    """fraction of interior corners within 90 +/- 25 deg"""
    n = len(P)
    good = tot = 0
    for i in range(n):
        v1 = P[i - 1] - P[i]; v2 = P[(i + 1) % n] - P[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(v1, v2)) / (n1 * n2)))))
        tot += 1
        if 65.0 <= ang <= 115.0:
            good += 1
    return good / max(tot, 1)


def despike(P):
    """iteratively drop vertices with corner angle < SPIKE_ANG"""
    P = np.array(P, float)
    for _ in range(12):
        removed = False
        out = []
        n = len(P)
        for i in range(n):
            v1 = P[i - 1] - P[i]; v2 = P[(i + 1) % n] - P[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(v1, v2)) / (n1 * n2)))))
                if ang < SPIKE_ANG:
                    removed = True
                    continue
            out.append(P[i])
        P = np.array(out)
        if len(P) < 4 or not removed:
            break
    return P


def regularize(contour, theta):
    """contour: (n,2) closed ring (m). Returns (poly, info) or (None, why)."""
    dp = measure.approximate_polygon(contour, tolerance=DP_TOL * CELL)
    if len(dp) < 4:
        return None, "too_few_vertices"
    if dp[0][0] == dp[-1][0] and dp[0][1] == dp[-1][1]:
        dp = dp[:-1]
    n = len(dp)
    # map dp vertices back to contour indices for support points
    idx = []
    j = 0
    for v in dp:
        while j < len(contour) and not (contour[j][0] == v[0] and contour[j][1] == v[1]):
            j += 1
        idx.append(min(j, len(contour) - 1))
    # per-segment class
    raw = []
    for i in range(n):
        a, b = dp[i], dp[(i + 1) % n]
        L = float(np.linalg.norm(b - a))
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        raw.append((ang, L, idx[i], idx[(i + 1) % n]))
    # City3D-style orientation clustering: primary axes from ridge theta,
    # plus secondary direction clusters from persistent non-primary edges
    cands = [theta, (theta + 90.0) % 180.0]
    clusters = []          # [sum_sin4, sum_cos4, weight]
    for ang, L, _, _ in sorted(raw, key=lambda t: -t[1]):
        if min(ang_dist(ang, c) for c in cands) <= SNAP_ANG:
            continue
        placed = False
        for cl in clusters:
            mean = (math.degrees(math.atan2(cl[0], cl[1])) / 4.0) % 180.0
            if min(ang_dist(ang, mean), ang_dist(ang, (mean + 90) % 180)) <= 10.0:
                a4 = math.radians(ang * 4.0)
                cl[0] += L * math.sin(a4); cl[1] += L * math.cos(a4); cl[2] += L
                placed = True
                break
        if not placed:
            a4 = math.radians(ang * 4.0)
            clusters.append([L * math.sin(a4), L * math.cos(a4), L])
    for cl in clusters:
        if cl[2] >= 3.0:
            mean = (math.degrees(math.atan2(cl[0], cl[1])) / 4.0) % 180.0
            cands.append(mean)
            cands.append((mean + 90.0) % 180.0)
    segs = []
    for ang, L, i0, i1 in raw:
        cls = None
        best = SNAP_ANG + 1
        for cand in cands:
            d = ang_dist(ang, cand)
            if d <= SNAP_ANG and d < best:
                cls = cand
                best = d
        segs.append(dict(ang=ang, cls=cls, L=L, i0=i0, i1=i1))
    # merge consecutive same-class segments
    merged = []
    for s in segs:
        if merged:
            p = merged[-1]
            same = (p["cls"] is not None and s["cls"] == p["cls"]) or \
                   (p["cls"] is None and s["cls"] is None and
                    ang_dist(p["ang"], s["ang"]) <= 8.0)
            if same:
                p["L"] += s["L"]
                p["i1"] = s["i1"]
                continue
        merged.append(dict(s))
    if len(merged) > 1:
        p, l = merged[0], merged[-1]
        same = (p["cls"] is not None and l["cls"] == p["cls"]) or \
               (p["cls"] is None and l["cls"] is None and
                ang_dist(p["ang"], l["ang"]) <= 8.0)
        if same:
            p["L"] += l["L"]
            p["i0"] = l["i0"]
            merged.pop()
    # absorb short runs into neighbors
    merged = [m for m in merged if m["L"] >= MIN_SEG] or merged
    if len(merged) < 4:
        return None, "too_few_edges"
    # fit line offset from support points
    lines = []
    for m in merged:
        i0, i1 = m["i0"], m["i1"]
        pts = contour[i0:i1 + 1] if i1 >= i0 else np.vstack([contour[i0:], contour[:i1 + 1]])
        phi = math.radians(m["cls"] if m["cls"] is not None else m["ang"])
        nvec = np.array([-math.sin(phi), math.cos(phi)])
        uvec = np.array([math.cos(phi), math.sin(phi)])
        d = float((pts @ nvec).mean())
        proj = lambda p: p - nvec * (float(np.dot(p, nvec)) - d)
        lines.append(dict(phi=phi, n=nvec, u=uvec, d=d, ref=pts.mean(axis=0),
                          p0=proj(pts[0]), p1=proj(pts[-1]),
                          snapped=m["cls"] is not None, L=m["L"]))
    # near-collinear consecutive line merge (kills micro Z-jogs)
    changed = True
    while changed and len(lines) > 4:
        changed = False
        for i in range(len(lines)):
            j = (i + 1) % len(lines)
            li, lj = lines[i], lines[j]
            if ang_dist(math.degrees(li["phi"]), math.degrees(lj["phi"])) >= LINE_MERGE_ANG:
                continue
            off = abs(float(np.dot(lj["ref"] - li["ref"], li["n"])))
            if off >= LINE_MERGE_OFF:
                continue
            w1, w2 = li["L"], lj["L"]
            phi = li["phi"] if w1 >= w2 else lj["phi"]
            nvec = np.array([-math.sin(phi), math.cos(phi)])
            d = (float(np.dot(li["ref"], nvec)) * w1 + float(np.dot(lj["ref"], nvec)) * w2) / (w1 + w2)
            proj = lambda p: p - nvec * (float(np.dot(p, nvec)) - d)
            merged_line = dict(phi=phi, n=nvec, u=np.array([math.cos(phi), math.sin(phi)]),
                               d=d, ref=(li["ref"] * w1 + lj["ref"] * w2) / (w1 + w2),
                               p0=proj(li["p0"]), p1=proj(lj["p1"]),
                               snapped=li["snapped"] or lj["snapped"], L=w1 + w2)
            if j > i:
                lines = lines[:i] + [merged_line] + lines[j + 1:]
            else:
                lines = [merged_line] + lines[1:i]
            changed = True
            break
    if len(lines) < 4:
        return None, "too_few_edges"

    # rebuild polygon
    poly = []
    n_l = len(lines)
    for i in range(n_l):
        li, lj = lines[i], lines[(i + 1) % n_l]
        dphi = ang_dist(math.degrees(li["phi"]), math.degrees(lj["phi"]))
        corner = None
        if dphi >= PAR_LIMIT:
            A = np.vstack([li["n"], lj["n"]])
            b = np.array([li["d"], lj["d"]])
            try:
                c0 = np.linalg.solve(A, b)
                # overshoot guard: corner must sit near both support ends
                if (np.linalg.norm(c0 - li["p1"]) <= OVERSHOOT and
                        np.linalg.norm(c0 - lj["p0"]) <= OVERSHOOT):
                    corner = c0
            except np.linalg.LinAlgError:
                pass
        if corner is not None:
            poly.append(corner)
        else:
            e_j = lj["p0"] - lj["n"] * 0.0
            poly.append(li["p1"]); poly.append(e_j)
    poly = np.array(poly)
    # spike & sliver cleanup: drop near-duplicate vertices, then iteratively
    # remove acute spike tips (corner angle < SPIKE_ANG with a short edge)
    for _ in range(12):
        keep = [0]
        for i in range(1, len(poly)):
            if np.linalg.norm(poly[i] - poly[keep[-1]]) > 0.3:
                keep.append(i)
        if len(keep) >= 2 and np.linalg.norm(poly[keep[-1]] - poly[keep[0]]) <= 0.3:
            keep.pop()
        poly = poly[keep]
        if len(poly) < 4:
            return None, "degenerate"
        removed = False
        out = []
        n_p = len(poly)
        for i in range(n_p):
            a, b2, c = poly[i - 1], poly[i], poly[(i + 1) % n_p]
            v1, v2 = a - b2, c - b2
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-6 and n2 > 1e-6:
                ang = math.degrees(math.acos(max(-1, min(1, float(np.dot(v1, v2)) / (n1 * n2)))))
                if ang < SPIKE_ANG:
                    removed = True
                    continue
            out.append(b2)
        poly = np.array(out)
        if len(poly) < 4:
            return None, "degenerate"
        if not removed:
            break
    if self_intersects(poly):
        return None, "self_intersect"
    ortho_share = sum(l["L"] for l in lines if l["snapped"]) / max(
        sum(l["L"] for l in lines), 1e-9)
    return poly, dict(ortho=ortho_share, n_edges=len(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    las = laspy.read(FULL / "buildings_planes.laz")
    inst = np.asarray(las.instance_id)
    styp = np.asarray(las.surf_type)
    pid = np.asarray(las.plane_id)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                         np.asarray(las.z)]) * FT_TO_M

    # ridge directions per building
    ridge_dir = {}
    with open(FULL / "cues.csv") as f:
        for r in csv.DictReader(f):
            if r["type"] not in ("ridge", "valley"):
                continue
            fid = int(r["instance_id"])
            dx = float(r["x2"]) - float(r["x1"])
            dy = float(r["y2"]) - float(r["y1"])
            L = float(r["length_m"])
            ang = math.degrees(math.atan2(dy, dx)) % 180.0
            ridge_dir.setdefault(fid, []).append((ang, L))
    zmin_roof = {}
    with open(FULL / "planes.csv") as f:
        for r in csv.DictReader(f):
            if r["type"] == "roof":
                fid = int(r["instance_id"])
                zmin_roof[fid] = min(zmin_roof.get(fid, 1e9), float(r["cz_m"]))

    order = np.argsort(inst, kind="stable")
    uniq, starts = np.unique(inst[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    rows = []
    render = {}
    stats = dict(ok=0, fallback=0)
    t0 = time.time()
    for fid, a, b in zip(uniq, starts, ends):
        if fid == 0:
            continue
        idx = order[a:b]
        P = X[idx]
        rm = (styp[idx] == 1) & (pid[idx] > 0)
        if rm.sum() < 12:
            continue
        x0m, y0m = P[:, 0].min(), P[:, 1].min()
        gx = ((P[rm, 0] - x0m) / CELL).astype(np.int32)
        gy = ((P[rm, 1] - y0m) / CELL).astype(np.int32)
        mm = np.zeros((gy.max() + 3, gx.max() + 3), np.uint16)
        np.add.at(mm, (gy + 1, gx + 1), 1)
        occ0 = mm >= 1
        s8 = np.ones((3, 3), bool)
        lab0, n0 = ndimage.label(occ0, structure=s8)
        L0 = np.bincount(lab0.ravel())[1:].max() if n0 else 0
        opened = ndimage.binary_opening(occ0, structure=s8)
        lab1, n1 = ndimage.label(opened, structure=s8)
        L1 = np.bincount(lab1.ravel())[1:].max() if n1 else 0
        # connectivity guard: largest component must survive (>=80%) and
        # total loss must stay under 20%
        base = opened if (L1 >= 0.8 * L0 and opened.sum() >= 0.8 * occ0.sum()) else occ0
        occ_cells = int(base.sum())
        m = ndimage.binary_fill_holes(
            ndimage.binary_closing(base, structure=s8))
        hole_ratio = 1.0 - occ_cells / max(int(m.sum()), 1)
        # satellites excluded from extent estimation (contour already takes
        # the main component implicitly)
        lab_m, n_m = ndimage.label(m, structure=s8)
        n_big = 1
        if n_m > 1:
            sizes = np.bincount(lab_m.ravel())[1:]
            big_ids = [i + 1 for i, s in enumerate(sizes)
                       if s >= 0.25 * sizes.max()]
            n_big = len(big_ids)
            base_rect = base & np.isin(lab_m, big_ids)
        else:
            base_rect = base
        conts = measure.find_contours(m.astype(float), 0.5)
        if not conts:
            continue
        cont = max(conts, key=len)
        cont_m = (cont[:, ::-1] - 1) * CELL  # local meters (origin x0m,y0m)
        # dominant direction
        rd = ridge_dir.get(int(fid), [])
        if sum(L for _, L in rd) >= 5.0:
            theta = fold_mean_dir([a0 for a0, _ in rd], [L for _, L in rd])
        else:
            dpv = measure.approximate_polygon(cont_m, tolerance=2 * CELL)
            angs, ws = [], []
            for i in range(len(dpv) - 1):
                d = dpv[i + 1] - dpv[i]
                L = np.linalg.norm(d)
                if L > 0.5:
                    angs.append(math.degrees(math.atan2(d[1], d[0])) % 180.0)
                    ws.append(L)
            theta = fold_mean_dir(angs, ws) if angs else 0.0
        poly, info = regularize(cont_m, theta)
        raw_area = poly_area(measure.approximate_polygon(cont_m, tolerance=CELL))
        mask_area = float(m.sum()) * CELL * CELL
        dens = rm.sum() / max(mask_area * 30.0, 1e-9)   # occlusion indicator
        # oriented minimum bounding rectangle at theta
        th = math.radians(theta)
        R = np.array([[math.cos(-th), -math.sin(-th)], [math.sin(-th), math.cos(-th)]])
        bys, bxs = np.nonzero(base_rect)
        cellpts = np.column_stack([(bxs - 1 + 0.5) * CELL, (bys - 1 + 0.5) * CELL])
        rotc = cellpts @ R.T
        lo = np.empty(2); hi = np.empty(2)
        for ax in (0, 1):
            v = rotc[:, ax]
            edges = np.arange(v.min() - CELL, v.max() + 2 * CELL, CELL)
            cnts, _ = np.histogram(v, bins=edges)
            nz = cnts[cnts > 0]
            thr = max(2.0, 0.10 * np.percentile(nz, 90))
            occ_idx = np.nonzero(cnts > 0)[0]
            i0, i1 = occ_idx[0], occ_idx[-1]
            total = cnts.sum()
            trimmed = 0
            while i0 < i1 and cnts[i0] < thr and trimmed + cnts[i0] <= 0.2 * total:
                trimmed += cnts[i0]; i0 += 1
            while i1 > i0 and cnts[i1] < thr and trimmed + cnts[i1] <= 0.2 * total:
                trimmed += cnts[i1]; i1 -= 1
            lo[ax] = edges[i0]
            hi[ax] = edges[i1 + 1]
        rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) @ R
        fill = raw_area / max(poly_area(rect), 1e-9)
        if n_big >= 2:   # split-by-occlusion: judge fill by union of big parts
            fill = (base_rect.sum() * CELL * CELL) / max(poly_area(rect), 1e-9)
        bad_shape = poly is None or \
            abs(poly_area(poly) - raw_area) / max(raw_area, 1e-9) > AREA_TOL
        ortho0 = 0.0 if (poly is None or isinstance(info, str)) else info["ortho"]
        n_edges0 = 0 if (poly is None or isinstance(info, str)) else info["n_edges"]
        creg = 1.0 if (poly is None or len(poly) < 4) else corner_regularity(poly)
        score = (int(hole_ratio > 0.25) + int(dens < 0.35)
                 + int(ortho0 < 0.45 and n_edges0 >= 8)
                 + 2 * int(creg < 0.6) + 2 * int(bad_shape)
                 + 2 * int(n_big >= 2))   # mask split by occlusion -> rect over union
        use_rect = score >= 2 and fill >= (0.50 if n_big >= 2 else 0.62)
        if use_rect:
            poly = rect
            ok = True
            why = ""
            ortho = 1.0
            method = "rect_prior"
            stats["rect"] = stats.get("rect", 0) + 1
        elif not bad_shape:
            ok = True
            why = ""
            ortho = info["ortho"]
            method = "regular"
            stats["ok"] += 1
        else:
            poly = measure.approximate_polygon(cont_m, tolerance=DP_TOL * CELL)
            if np.allclose(poly[0], poly[-1]):
                poly = poly[:-1]
            poly = despike(poly)
            ok = False
            why = info if isinstance(info, str) else "area_change"
            ortho = 0.0
            method = "dp_fallback"
            stats["fallback"] += 1
        # mean offset raw->poly (sampled)
        samp = cont_m[::max(1, len(cont_m) // 80)]
        dmin = np.full(len(samp), np.inf)
        for i in range(len(poly)):
            a2, b2 = poly[i], poly[(i + 1) % len(poly)]
            ab = b2 - a2
            t = np.clip(((samp - a2) @ ab) / max(ab @ ab, 1e-9), 0, 1)
            proj = a2 + t[:, None] * ab
            dmin = np.minimum(dmin, np.linalg.norm(samp - proj, axis=1))
        wkt = "POLYGON((" + ",".join(
            f"{(x+x0m)/FT_TO_M:.2f} {(y+y0m)/FT_TO_M:.2f}" for x, y in poly) + "))"
        rows.append((int(fid), len(poly), round(poly_area(poly), 1),
                     round(float(theta), 1), round(ortho, 3),
                     round(float(dmin.mean()), 2), int(ok), why, method,
                     round(dens, 2), round(fill, 2), round(hole_ratio, 2), wkt))
        if int(fid) in PILOT or (not ok and len(render) < 24):
            render[int(fid)] = (cont_m, poly, theta, ok)
    log(f"regularized {stats['ok']} ok, {stats.get('rect',0)} rect_prior, {stats['fallback']} fallback "
        f"({time.time()-t0:.0f}s)")

    with open(OUT / "outlines_regularized.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "n_vertices", "area_m2", "theta_deg",
                    "ortho_share", "mean_offset_m", "regularized", "fallback_reason",
                    "method", "density_ratio", "rect_fill", "hole_ratio", "wkt_ftus"])
        w.writerows(rows)

    with open(OUT / "outlines_regularized.obj", "w") as f:
        f.write("# regularized outlines at eave height, ftUS CRS\n")
        v = 1
        for r in rows:
            fid = r[0]
            z = zmin_roof.get(fid)
            if z is None:
                continue
            zf = z / FT_TO_M
            coords = r[12].split("((")[1].rstrip("))").split(",")
            ids = []
            for cstr in coords:
                xs, ys = cstr.split()
                f.write(f"v {xs} {ys} {zf:.2f}\n")
                ids.append(v); v += 1
            f.write("l " + " ".join(map(str, ids + [ids[0]])) + "\n")

    # ---------------- QA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [f for f in PILOT if f in render] + \
           [f for f in render if f not in PILOT][:6]
    ncol = 5
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.9 * nrow))
    axes = np.atleast_2d(axes)
    for k, fid in enumerate(show):
        ax = axes[k // ncol, k % ncol]
        cont_m, poly, theta, ok = render[fid]
        c0 = cont_m.mean(axis=0)
        ax.plot(cont_m[:, 0] - c0[0], cont_m[:, 1] - c0[1], "-", c="0.7", lw=1)
        pp = np.vstack([poly, poly[:1]])
        ax.plot(pp[:, 0] - c0[0], pp[:, 1] - c0[1], "r-" if ok else "b-", lw=1.8)
        ax.set_title(f"id {fid} | θ={theta:.0f}° | {len(poly)} vtx"
                     f"{'' if ok else ' | FALLBACK'}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(len(show), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("Outline regularization: gray=raw contour, red=regularized "
                 "(blue=fallback)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "qa_outline_reg.png", dpi=140)

    va = np.array([r[1] for r in rows])
    off = np.array([r[5] for r in rows])
    osh = np.array([r[4] for r in rows if r[6] == 1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(np.clip(va, 0, 40), bins=40, color="#4477aa")
    axes[0].set_title("vertices per outline")
    axes[1].hist(np.clip(off, 0, 1.5), bins=40, color="#4477aa")
    axes[1].set_title("mean offset raw->regularized (m)")
    axes[2].hist(osh, bins=40, color="#4477aa")
    axes[2].set_title("orthogonal-edge share (regularized only)")
    fig.tight_layout()
    fig.savefig(OUT / "qa_outline_metrics.png", dpi=140)
    log(f"median vertices {np.median(va):.0f}, median offset {np.median(off):.2f} m, "
        f"median ortho share {np.median(osh):.2f}")
    log("done")


if __name__ == "__main__":
    main()
