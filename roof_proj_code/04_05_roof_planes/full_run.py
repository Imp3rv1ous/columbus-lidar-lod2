#!/usr/bin/env python
"""Full-scale roof-plane extraction: all kept building instances.

Outputs (output_full/):
  buildings_planes.laz  clean points + plane_id (global, 0=unassigned),
                        surf_type (0 none / 1 roof / 2 wall); instance_id,
                        RGB etc. carried over from buildings_clean.laz
  planes.csv            one row per plane: ids, type, points, projected area,
                        normal, centroid, rms, slope, azimuth
  building_quality.csv  Gate-C table: per building plane counts, assigned
                        ratio, rms, roof-form heuristic, flags
  qa_gallery.png        24 stratified random buildings colored by plane
  qa_summary.png        distributions: planes/building, assigned ratio, forms
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
from plane_extraction import segment, WALL_NZ, MIN_PTS

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
CLEAN = BASE / "02_03_instance_seg/output_v2/buildings_clean.laz"
OUT = BASE / "04_05_roof_planes/output_full"
FT_TO_M = 0.30480060960121924


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def roof_form(roof_planes, roof_pts_total):
    """heuristic roof-form label from significant roof planes"""
    if not roof_planes:
        return "none"
    sig = [(p, n) for p, n in roof_planes
           if n >= max(80, 0.08 * roof_pts_total)]
    if not sig:
        return "fragmented"
    share0 = max(n for _, n in sig) / max(roof_pts_total, 1)
    slopes = [np.degrees(np.arccos(min(abs(p[0][2]), 1.0))) for p, _ in sig]
    azis = [np.degrees(np.arctan2(p[0][1], p[0][0])) for p, _ in sig]
    dom_slope = slopes[int(np.argmax([n for _, n in sig]))]
    if dom_slope < 10 and share0 >= 0.6:
        return "flat"
    sloped = [i for i, s in enumerate(slopes) if 10 <= s <= 55]
    if len(sig) == 1:
        return "shed" if 10 <= dom_slope <= 55 else "flat"
    if len(sig) == 2 and len(sloped) == 2:
        d = abs(((azis[0] - azis[1] + 180) % 360) - 180)
        if d > 135 and abs(slopes[0] - slopes[1]) < 12:
            return "gable"
        return "complex2"
    if 3 <= len(sig) <= 5 and len(sloped) >= 3:
        return "hip_or_pyramid"
    return "complex"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    las = laspy.read(CLEAN)
    inst = np.asarray(las.instance_id)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                         np.asarray(las.z)]) * FT_TO_M
    n_all = len(inst)
    plane_id_pt = np.zeros(n_all, np.uint32)
    surf_type_pt = np.zeros(n_all, np.uint8)

    order = np.argsort(inst, kind="stable")
    uniq, starts = np.unique(inst[order], return_index=True)
    ends = np.append(starts[1:], n_all)

    plane_rows = []
    qual_rows = []
    next_gid = 1
    t0 = time.time()
    done = 0
    n_inst = int((uniq > 0).sum())
    for fid, a, b in zip(uniq, starts, ends):
        if fid == 0:
            continue
        idx = order[a:b]
        P = X[idx]
        t1 = time.time()
        if len(P) < 25:
            labels = np.full(len(P), -1, np.int32)
            planes = {}
        else:
            labels, planes = segment(P)
        dt = time.time() - t1
        roof, wall = {}, {}
        for l, p in planes.items():
            (roof if abs(p[0][2]) >= WALL_NZ else wall)[l] = p
        roof_pts = int(sum(p[3] for p in roof.values()))
        for l in sorted(planes):
            nv, c, rms, npts = planes[l]
            sel = labels == l
            gx = ((P[sel, 0] - P[sel, 0].min()) / 0.5).astype(np.int64)
            gy = ((P[sel, 1] - P[sel, 1].min()) / 0.5).astype(np.int64)
            area = len(np.unique(gy * 100000 + gx)) * 0.25
            is_roof = l in roof
            slope = float(np.degrees(np.arccos(min(abs(nv[2]), 1.0))))
            azi = float(np.degrees(np.arctan2(nv[1], nv[0]))) if slope >= 5 else float("nan")
            plane_rows.append((next_gid, int(fid), int(l),
                               "roof" if is_roof else "wall", npts,
                               round(area, 1),
                               round(nv[0], 4), round(nv[1], 4), round(nv[2], 4),
                               round(c[0], 2), round(c[1], 2), round(c[2], 2),
                               round(rms, 4), round(slope, 1),
                               round(azi, 1) if azi == azi else ""))
            plane_id_pt[idx[sel]] = next_gid
            surf_type_pt[idx[sel]] = 1 if is_roof else 2
            next_gid += 1
        assigned = float((labels >= 0).mean()) if len(P) else 0.0
        form = roof_form([(p, p[3]) for p in roof.values()], roof_pts)
        mean_rms = float(np.mean([p[2] for p in roof.values()])) if roof else 0.0
        flags = []
        if assigned < 0.70:
            flags.append("low_assigned")
        if not roof:
            flags.append("no_roof_plane")
        qual_rows.append((int(fid), len(P), len(roof), len(wall),
                          round(assigned, 3), round(roof_pts / max(len(P), 1), 3),
                          round(mean_rms * 100, 1), form, ";".join(flags),
                          round(dt, 2)))
        done += 1
        if done % 200 == 0:
            log(f"{done}/{n_inst} buildings, {next_gid-1} planes, "
                f"{time.time()-t0:.0f}s elapsed")

    log(f"done segmenting: {done} buildings, {next_gid-1} planes, "
        f"{time.time()-t0:.0f}s")

    with open(OUT / "planes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plane_id", "instance_id", "local_id", "type", "n_points",
                    "area_m2", "nx", "ny", "nz", "cx_m", "cy_m", "cz_m",
                    "rms_m", "slope_deg", "azimuth_deg"])
        w.writerows(plane_rows)
    with open(OUT / "building_quality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "n_points", "n_roof_planes", "n_wall_planes",
                    "assigned_ratio", "roof_pt_ratio", "mean_rms_cm",
                    "roof_form", "flags", "seconds"])
        w.writerows(qual_rows)

    las.add_extra_dim(laspy.ExtraBytesParams(name="plane_id", type=np.uint32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="surf_type", type=np.uint8))
    las.plane_id = plane_id_pt
    las.surf_type = surf_type_pt
    las.write(OUT / "buildings_planes.laz")
    log("wrote buildings_planes.laz")

    # ---------------------------------------------------------- QA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    cmap = colormaps["tab20"]

    q = {r[0]: r for r in qual_rows}
    areas = {int(r[0]): float(r[1]) for r in qual_rows}
    rng = np.random.default_rng(3)
    strata = [(200, 2000), (2000, 4000), (4000, 8000), (8000, 30000),
              (30000, 10**9)]
    ids_by_pts = {}
    gallery = []
    all_ids = [r[0] for r in qual_rows]
    for lo, hi in strata:
        pool = [i for i in all_ids if lo <= q[i][1] < hi]
        take = min(5 if lo > 200 else 4, len(pool))
        gallery += list(rng.choice(pool, size=take, replace=False))
    gallery = gallery[:24]
    ncol, nrow = 6, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.8 * nrow))
    pos_of = {int(f): (s, e) for f, s, e in zip(uniq, starts, ends)}
    for k, fid in enumerate(gallery):
        ax = axes[k // ncol, k % ncol]
        a, b = pos_of[int(fid)]
        idx = order[a:b]
        P = X[idx]
        pid = plane_id_pt[idx]
        st = surf_type_pt[idx]
        un = pid == 0
        ax.scatter(P[un, 0] - P[:, 0].min(), P[un, 1] - P[:, 1].min(),
                   s=0.4, c="0.85", rasterized=True)
        for i, g in enumerate(np.unique(pid[pid > 0])):
            sel = pid == g
            col = "k" if st[sel][0] == 2 else cmap(i % 20)
            ax.scatter(P[sel, 0] - P[:, 0].min(), P[sel, 1] - P[:, 1].min(),
                       s=0.5, color=col, rasterized=True)
        r = q[int(fid)]
        ax.set_title(f"id {fid} | {r[2]}rf/{r[3]}w | {r[4]*100:.0f}% | {r[7]}",
                     fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "qa_gallery.png", dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nrf = [r[2] for r in qual_rows]
    axes[0].hist(np.clip(nrf, 0, 30), bins=31, color="#4477aa")
    axes[0].set_title("roof planes per building"); axes[0].set_xlabel("count")
    asr = [r[4] for r in qual_rows]
    axes[1].hist(asr, bins=40, color="#4477aa")
    axes[1].axvline(0.7, color="r", ls="--")
    axes[1].set_title("assigned-point ratio")
    forms = {}
    for r in qual_rows:
        forms[r[7]] = forms.get(r[7], 0) + 1
    ks = sorted(forms, key=lambda k: -forms[k])
    axes[2].bar(range(len(ks)), [forms[k] for k in ks], color="#4477aa")
    axes[2].set_xticks(range(len(ks)))
    axes[2].set_xticklabels(ks, rotation=30, ha="right")
    axes[2].set_title("roof form (heuristic)")
    fig.tight_layout()
    fig.savefig(OUT / "qa_summary.png", dpi=140)

    nrf = np.array(nrf); asr = np.array(asr)
    log(f"SUMMARY: buildings {done}, total planes {next_gid-1} "
        f"(roof {sum(1 for r in plane_rows if r[3]=='roof')}, "
        f"wall {sum(1 for r in plane_rows if r[3]=='wall')})")
    log(f"roof planes/building: median {np.median(nrf):.0f}, p90 {np.percentile(nrf,90):.0f}")
    log(f"assigned ratio: mean {asr.mean()*100:.1f}%, <70%: {(asr<0.7).sum()} buildings")
    log(f"roof forms: {dict(sorted(forms.items(), key=lambda x:-x[1]))}")
    log("done")


if __name__ == "__main__":
    main()
