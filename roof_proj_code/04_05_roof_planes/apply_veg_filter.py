#!/usr/bin/env python
"""Apply the reviewed vegetation-filter verdicts.

- WHITELIST: instances visually confirmed as real structures during review;
  their planes are never killed and they are never removed.
- Kill planes (plane_veg.csv kill=1, minus whitelist) -> points keep their
  instance but lose plane membership (plane_id=0, surf_type=0).
- Kill instances (instance_veg.csv kill_candidate=1, minus whitelist) ->
  points removed from buildings_planes.laz and buildings_clean.laz, saved
  to 02_03_instance_seg/output_v2/veg_reclassified.laz, and logged in
  instances_gateA.csv as reject_vegetation_retro (Gate A retro-verdict
  using class-5 context + stage-4 planarity feedback).
- planes.csv / building_quality.csv updated consistently (per-building
  counts, assigned ratio, mean rms and roof form recomputed).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).parent))
from full_run import roof_form
from plane_extraction import WALL_NZ

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
FULL = BASE / "04_05_roof_planes/output_full"
ISEG = BASE / "02_03_instance_seg/output_v2"

WHITELIST = {1743, 1745, 1987, 2147, 2000, 2163}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    kill_inst = set()
    for r in csv.DictReader(open(FULL / "instance_veg.csv")):
        if r["kill_candidate"] == "1" and int(r["instance_id"]) not in WHITELIST:
            kill_inst.add(int(r["instance_id"]))
    kill_planes = set()
    for r in csv.DictReader(open(FULL / "plane_veg.csv")):
        fid = int(r["instance_id"])
        if r["kill"] == "1" and fid not in WHITELIST and fid not in kill_inst:
            kill_planes.add(int(r["plane_id"]))
    log(f"verdicts: {len(kill_inst)} instances, {len(kill_planes)} planes")

    # ------------------------------------------------ buildings_planes.laz
    las = laspy.read(FULL / "buildings_planes.laz")
    inst = np.asarray(las.instance_id)
    pid = np.asarray(las.plane_id)
    rm_inst = np.isin(inst, np.array(sorted(kill_inst), dtype=np.uint32))
    rm_plane = np.isin(pid, np.array(sorted(kill_planes), dtype=np.uint32))
    las.plane_id[rm_plane] = 0
    las.surf_type[rm_plane] = 0
    veg = laspy.LasData(las.header)
    veg.points = las.points[rm_inst].copy()
    veg.write(ISEG / "veg_reclassified.laz")
    keep = laspy.LasData(las.header)
    keep.points = las.points[~rm_inst].copy()
    keep.write(FULL / "buildings_planes.laz")
    log(f"planes LAZ: removed {rm_inst.sum():,} instance pts, "
        f"cleared {rm_plane.sum():,} plane pts")

    # ------------------------------------------------ buildings_clean.laz
    cl = laspy.read(ISEG / "buildings_clean.laz")
    ci = np.asarray(cl.instance_id)
    m = ~np.isin(ci, np.array(sorted(kill_inst), dtype=np.uint32))
    out = laspy.LasData(cl.header)
    out.points = cl.points[m].copy()
    out.write(ISEG / "buildings_clean.laz")
    log(f"clean LAZ: {m.sum():,} pts kept")

    # ------------------------------------------------ gateA csv
    rows = list(csv.DictReader(open(ISEG / "instances_gateA.csv")))
    for r in rows:
        if int(r["instance_id"]) in kill_inst:
            r["decision"] = "reject_vegetation_retro"
    with open(ISEG / "instances_gateA.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # ------------------------------------------------ planes.csv
    prows = list(csv.DictReader(open(FULL / "planes.csv")))
    prows = [r for r in prows
             if int(r["plane_id"]) not in kill_planes
             and int(r["instance_id"]) not in kill_inst]
    with open(FULL / "planes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=prows[0].keys())
        w.writeheader(); w.writerows(prows)
    log(f"planes.csv: {len(prows)} planes remain")

    # ------------------------------------------------ building_quality.csv
    inst2 = np.asarray(keep.instance_id)
    pid2 = np.asarray(keep.plane_id)
    order = np.argsort(inst2, kind="stable")
    u, s = np.unique(inst2[order], return_index=True)
    e = np.append(s[1:], len(order))
    assigned = {}
    npts_of = {}
    for fid, a, b in zip(u, s, e):
        if fid == 0:
            continue
        sl = order[a:b]
        npts_of[int(fid)] = len(sl)
        assigned[int(fid)] = float((pid2[sl] > 0).mean())
    by_inst = {}
    for r in prows:
        by_inst.setdefault(int(r["instance_id"]), []).append(r)
    qrows = list(csv.DictReader(open(FULL / "building_quality.csv")))
    out_q = []
    for r in qrows:
        fid = int(r["instance_id"])
        if fid in kill_inst:
            continue
        pls = by_inst.get(fid, [])
        roof = [p for p in pls if p["type"] == "roof"]
        wall = [p for p in pls if p["type"] == "wall"]
        roof_pts = sum(int(p["n_points"]) for p in roof)
        r["n_roof_planes"] = len(roof)
        r["n_wall_planes"] = len(wall)
        r["assigned_ratio"] = round(assigned.get(fid, 0.0), 3)
        r["roof_pt_ratio"] = round(roof_pts / max(npts_of.get(fid, 1), 1), 3)
        r["mean_rms_cm"] = round(float(np.mean([float(p["rms_m"]) for p in roof])) * 100, 1) if roof else 0.0
        rp = [([np.array([float(p["nx"]), float(p["ny"]), float(p["nz"])]),
                None, float(p["rms_m"]), int(p["n_points"])], int(p["n_points"]))
              for p in roof]
        r["roof_form"] = roof_form(rp, roof_pts)
        flags = []
        if float(r["assigned_ratio"]) < 0.70:
            flags.append("low_assigned")
        if not roof:
            flags.append("no_roof_plane")
        r["flags"] = ";".join(flags)
        out_q.append(r)
    with open(FULL / "building_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=qrows[0].keys())
        w.writeheader(); w.writerows(out_q)
    log(f"building_quality.csv: {len(out_q)} buildings; "
        f"low_assigned: {sum(1 for r in out_q if 'low_assigned' in r['flags'])}")
    log("done — regenerate recolor/cues/surfaces next")


if __name__ == "__main__":
    main()
