#!/usr/bin/env python
"""Audit + minimal fix for the instance-id 2296 reuse in instances_gateA.csv.

Incident (2026-08-30): 06_outline_reg/apply_splits.py allocated new child
ids from max(buildings_clean.laz instance_id)+1.  The retro vegetation
rejection had already removed instance 2296 from buildings_clean.laz, so
the max was 2295 and child ids started at 2296 -- colliding with the
retired vegetation instance.  instances_gateA.csv therefore holds TWO
rows with instance_id=2296: the retired vegetation one
(reject_vegetation_retro) and the split child of 1244
(keep_split_child_of_1244).  Every live downstream file (buildings_clean
/ buildings_planes / planes / cues / edges / building_quality /
outlines) was produced AFTER the split, so 2296 there means the child.

Modes:
  --assess   read-only impact assessment: for every instance-keyed file,
             locate id 2296 and classify OLD (vegetation) vs NEW (child)
             by centroid / point count evidence.  Prints a table and
             writes nothing.
  --fix      minimal fix, audit-preserving (no row deleted):
             1. byte-exact backup instances_gateA.csv ->
                instances_gateA_pre_idfix_20260830.bak
             2. rewrite instances_gateA.csv with a new `note` column
                (empty for unaffected rows); the retired vegetation row
                gets instance_id=2296_retired +
                note=id_retired_reused_by_split_child_of_1244, the child
                row gets note=id_reused_from_retired_vegetation_instance;
                additionally rows 2306/2307 (children of 228) get their
                n_points corrected from the parent total 7301 to the
                per-child counts measured in buildings_clean.laz
                (3419/3882, sum=7301), noted with
                n_points_corrected_from_buildings_clean_was_7301
             3. retrofill the missing 228 -> 2306;2307 row in
                06_outline_reg/output/split_applied.csv
                (ground_frac=unlogged_retrofill; verdict source is
                split_verdicts.csv ground_frac_in_gap=1.0)
             Refuses to run twice (detects 2296_retired / note column).

Run check_id_consistency.py afterwards for the cross-file id-set
reconciliation.
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import laspy

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
ISEG = BASE / "02_03_instance_seg/output_v2"
ROOF = BASE / "04_05_roof_planes/output_full"
OREG = BASE / "06_outline_reg/output"
FT_TO_M = 0.30480060960121924

GATEA = ISEG / "instances_gateA.csv"
BACKUP = ISEG / "instances_gateA_pre_idfix_20260830.bak"

# gateA cx_m/cy_m are LOCAL metres; instances.csv cx_m/cy_m are absolute
# NAD83 metres.  Derive the constant local->absolute offset by joining
# the two tables on the 2,296 original ids, then place both 2296
# candidates in absolute coordinates for evidence comparison.
OLD_LOCAL = np.array([26.4, 588.1])     # retired vegetation row, 425 pts
NEW_LOCAL = np.array([1354.6, 533.6])   # child row (inherits parent 1244
                                        # centroid; child is ~2 m off)
CHILD_IDS = list(range(2296, 2308))     # the 12 split children


def local_origin():
    inst_abs = {int(r["instance_id"]): (float(r["cx_m"]), float(r["cy_m"]))
                for r in csv.DictReader(open(ISEG / "instances.csv"))}
    dx, dy = [], []
    for r in csv.DictReader(open(GATEA)):
        rid = r["instance_id"].replace("_retired", "")
        if not rid.isdigit() or int(rid) in CHILD_IDS:
            continue
        ax, ay = inst_abs[int(rid)]
        dx.append(ax - float(r["cx_m"]))
        dy.append(ay - float(r["cy_m"]))
    ox, oy = float(np.median(dx)), float(np.median(dy))
    assert np.percentile(np.abs(np.array(dx) - ox), 99) < 1.0
    assert np.percentile(np.abs(np.array(dy) - oy), 99) < 1.0
    return np.array([ox, oy])


ORIGIN = None  # filled in assess()


def classify(cx, cy):
    """cx, cy in ABSOLUTE metres."""
    d_old = float(np.hypot(*((ORIGIN + OLD_LOCAL) - (cx, cy))))
    d_new = float(np.hypot(*((ORIGIN + NEW_LOCAL) - (cx, cy))))
    which = "OLD(veg)" if d_old < d_new else "NEW(child_of_1244)"
    return which, d_old, d_new


def laz_instance_stats(path, fid):
    las = laspy.read(path)
    inst = np.asarray(las.instance_id)
    m = inst == fid
    if not m.any():
        return None
    x = np.asarray(las.x)[m] * FT_TO_M
    y = np.asarray(las.y)[m] * FT_TO_M
    return int(m.sum()), float(x.mean()), float(y.mean())


def wkt_centroid_m(wkt):
    body = wkt[wkt.index("((") + 2:wkt.index("))")]
    pts = np.array([[float(v) for v in p.split()] for p in body.split(",")])
    return pts[:, 0].mean() * FT_TO_M, pts[:, 1].mean() * FT_TO_M


def assess():
    global ORIGIN
    ORIGIN = local_origin()
    rows = []  # (file, era, evidence, verdict)

    gate = list(csv.DictReader(open(GATEA)))
    dup = [r for r in gate if r["instance_id"] in ("2296", "2296_retired")]
    print(f"instances_gateA.csv rows with id 2296: {len(dup)}")
    for r in dup:
        print(f"  n_points={r['n_points']} area={r['area_m2']} "
              f"c=({r['cx_m']},{r['cy_m']}) decision={r['decision']}")
    print(f"gateA local->absolute origin: ({ORIGIN[0]:.2f}, {ORIGIN[1]:.2f}) m")
    print(f"  OLD veg 2296 absolute: ({ORIGIN[0]+OLD_LOCAL[0]:.1f}, "
          f"{ORIGIN[1]+OLD_LOCAL[1]:.1f})   "
          f"NEW child 2296 (parent centroid) absolute: "
          f"({ORIGIN[0]+NEW_LOCAL[0]:.1f}, {ORIGIN[1]+NEW_LOCAL[1]:.1f})")

    for name, path in [("buildings_clean.laz", ISEG / "buildings_clean.laz"),
                       ("buildings_planes.laz", ROOF / "buildings_planes.laz"),
                       ("buildings_instances.laz",
                        ISEG / "buildings_instances.laz"),
                       ("noise_removed.laz", ISEG / "noise_removed.laz")]:
        era = {"buildings_clean.laz": "post-split",
               "buildings_planes.laz": "post-split",
               "buildings_instances.laz": "pre-gateA",
               "noise_removed.laz": "gateA-era"}[name]
        st = laz_instance_stats(path, 2296)
        if st is None:
            rows.append((name, era, "id 2296 absent", "-"))
            continue
        n, cx, cy = st
        which, d_old, d_new = classify(cx, cy)
        rows.append((name, era,
                     f"{n} pts, centroid=({cx:.1f},{cy:.1f}) m "
                     f"d_old={d_old:.1f} d_new={d_new:.1f}", which))

    # veg_reclassified.laz: points of the 22 retro-rejected instances
    try:
        las = laspy.read(ISEG / "veg_reclassified.laz")
        inst = np.asarray(las.instance_id)
        m = inst == 2296
        if m.any():
            x = np.asarray(las.x)[m] * FT_TO_M
            y = np.asarray(las.y)[m] * FT_TO_M
            which, d_old, d_new = classify(x.mean(), y.mean())
            rows.append(("veg_reclassified.laz", "pre-split",
                         f"{int(m.sum())} pts, centroid="
                         f"({x.mean():.1f},{y.mean():.1f}) m", which))
        else:
            rows.append(("veg_reclassified.laz", "pre-split",
                         "id 2296 absent", "-"))
    except Exception as e:  # no instance_id dim etc.
        rows.append(("veg_reclassified.laz", "pre-split", f"unreadable: {e}", "?"))

    pl = [r for r in csv.DictReader(open(ROOF / "planes.csv"))
          if r["instance_id"] == "2296"]
    if pl:
        cx = np.mean([float(r["cx_m"]) for r in pl])
        cy = np.mean([float(r["cy_m"]) for r in pl])
        which, d_old, d_new = classify(cx, cy)
        rows.append(("planes.csv", "post-split",
                     f"{len(pl)} planes, mean plane centroid="
                     f"({cx:.1f},{cy:.1f}) m", which))
    else:
        rows.append(("planes.csv", "post-split", "id 2296 absent", "-"))

    bq = [r for r in csv.DictReader(open(ROOF / "building_quality.csv"))
          if r["instance_id"] == "2296"]
    rows.append(("building_quality.csv", "post-split",
                 f"{len(bq)} row(s), n_points="
                 f"{bq[0]['n_points'] if bq else '-'}",
                 "NEW(child_of_1244)" if bq and bq[0]["n_points"] == "1154"
                 else ("?" if bq else "-")))

    cue_hits = [r for r in csv.DictReader(open(ROOF / "cues.csv"))
                if r["instance_id"] == "2296"]
    if cue_hits:
        cx = np.mean([float(r["x1"]) for r in cue_hits])
        cy = np.mean([float(r["y1"]) for r in cue_hits])
        which, d_old, d_new = classify(cx, cy)
        rows.append(("cues.csv", "post-split",
                     f"{len(cue_hits)} cue(s), mean endpoint="
                     f"({cx:.1f},{cy:.1f}) m", which))
    else:
        rows.append(("cues.csv", "post-split", "id 2296 absent", "-"))
    n_e = sum(1 for r in csv.DictReader(open(ROOF / "edges.csv"))
              if r["instance_id"] == "2296")
    rows.append(("edges.csv", "post-split", f"{n_e} edge rows for id 2296",
                 "keyed to current planes.csv" if n_e else "-"))

    for name, path in [("outlines.csv", ROOF / "outlines.csv"),
                       ("outlines_regularized.csv",
                        OREG / "outlines_regularized.csv")]:
        ol = [r for r in csv.DictReader(open(path))
              if r["instance_id"] == "2296"]
        if ol:
            cx, cy = wkt_centroid_m(ol[0]["wkt_ftus"])
            which, d_old, d_new = classify(cx, cy)
            rows.append((name, "post-split",
                         f"WKT centroid=({cx:.1f},{cy:.1f}) m", which))
        else:
            rows.append((name, "post-split", "id 2296 absent", "-"))

    # pre-split-era audit tables: 2296 there is the OLD vegetation instance
    for name, path, era in [
            ("instances.csv", ISEG / "instances.csv", "pre-gateA"),
            ("shape_features.csv", ISEG / "shape_features.csv", "pre-split"),
            ("instance_veg.csv", ROOF / "instance_veg.csv", "pre-split"),
            ("plane_veg.csv", ROOF / "plane_veg.csv", "pre-split")]:
        hit = [r for r in csv.DictReader(open(path))
               if r.get("instance_id") == "2296"]
        if not hit:
            rows.append((name, era, "id 2296 absent", "-"))
            continue
        ev = f"{len(hit)} row(s)"
        if "cx_m" in hit[0]:  # absolute metres in instances.csv
            which, _, _ = classify(float(hit[0]["cx_m"]), float(hit[0]["cy_m"]))
            ev += (f", centroid=({float(hit[0]['cx_m']):.1f},"
                   f"{float(hit[0]['cy_m']):.1f}) m")
        elif name == "instance_veg.csv":
            which = ("OLD(veg)" if hit[0]["n_points"] == "425"
                     else "?")
            ev += f", n_points={hit[0]['n_points']} (old veg had 425)"
        elif name == "plane_veg.csv":
            tot = sum(int(r["n_points"]) for r in hit)
            which = ("OLD(veg) by era" if tot <= 425 else "?")
            ev += (f", sum plane pts={tot} <= 425; plane_ids join the "
                   f"pre-split planes generation, NOT current planes.csv")
        else:
            which = "OLD(veg) by era"
        rows.append((name, era, ev, which))

    # post-split iteration snapshots of the regularizer
    for name in ("_before_robust.csv", "_before_spikefix.csv"):
        try:
            hit = [r for r in csv.DictReader(open(OREG / name))
                   if r.get("instance_id") == "2296"]
            ev = f"{len(hit)} row(s) for id 2296"
            rows.append((name, "post-split", ev,
                         "NEW(child_of_1244)" if hit else "-"))
        except FileNotFoundError:
            pass

    # 02_03 watershed splits_log (instance-seg-internal, pre-gateA ids)
    txt = (ISEG / "splits_log.csv").read_text()
    n_mention = sum(1 for line in txt.splitlines()
                    if "2296" in line.split(",")[:2])
    rows.append(("splits_log.csv", "pre-gateA",
                 f"{n_mention} mention(s) of 2296 in id columns",
                 "OLD(veg) by era" if n_mention else "-"))

    print("\n=== impact table: what does instance_id=2296 mean in each file ===")
    w = max(len(r[0]) for r in rows)
    for name, era, ev, verdict in rows:
        print(f"{name:<{w}}  [{era:<10}]  {verdict:<20}  {ev}")

    # n_points crosscheck for all 12 children: gateA rows copied parent
    # geometry; n_points was overwritten per child -- verify vs laz
    print("\n=== split children n_points: gateA vs buildings_clean.laz ===")
    las = laspy.read(ISEG / "buildings_clean.laz")
    inst = np.asarray(las.instance_id)
    counts = dict(zip(*np.unique(inst, return_counts=True)))
    gate_np = {}
    for r in gate:
        if r["decision"].startswith("keep_split_child"):
            gate_np[int(r["instance_id"].replace("_retired", ""))] = (
                int(r["n_points"]), r["decision"])
    for fid in CHILD_IDS:
        gn = gate_np.get(fid, ("-", "-"))
        ln = counts.get(fid, 0)
        flag = "OK" if gn[0] == ln else "MISMATCH"
        print(f"  {fid}: gateA={gn[0]}  laz={ln}  {flag}   ({gn[1]})")
    return rows


def fix():
    gate = list(csv.DictReader(open(GATEA)))
    if any(r["instance_id"] == "2296_retired" for r in gate) or \
            "note" in gate[0]:
        sys.exit("fix already applied (2296_retired / note column present) "
                 "-- refusing to run twice")
    dup = [r for r in gate if r["instance_id"] == "2296"]
    assert len(dup) == 2, f"expected 2 rows with id 2296, got {len(dup)}"

    shutil.copy2(GATEA, BACKUP)
    print(f"backup written: {BACKUP.name}")

    # measured truth for the 228 children whose gateA n_points both got
    # the parent total (7301) written in by the applying run
    las = laspy.read(ISEG / "buildings_clean.laz")
    inst = np.asarray(las.instance_id)
    laz_np = {fid: int((inst == fid).sum()) for fid in (2306, 2307)}
    assert sum(laz_np.values()) == 7301, laz_np

    n_note = 0
    for r in gate:
        r["note"] = ""
        if r["instance_id"] == "2296":
            if r["decision"] == "reject_vegetation_retro":
                r["instance_id"] = "2296_retired"
                r["note"] = "id_retired_reused_by_split_child_of_1244"
                n_note += 1
            elif r["decision"] == "keep_split_child_of_1244":
                r["note"] = "id_reused_from_retired_vegetation_instance"
                n_note += 1
        elif r["instance_id"] in ("2306", "2307"):
            assert r["n_points"] == "7301"
            r["n_points"] = str(laz_np[int(r["instance_id"])])
            r["note"] = "n_points_corrected_from_buildings_clean_was_7301"
            n_note += 1
    assert n_note == 4
    fields = list(gate[0].keys())  # note appended last
    with open(GATEA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(gate)
    print(f"rewrote {GATEA.name}: {len(gate)} rows (unchanged), "
          f"1 id cell tombstoned, note column added ({n_note} rows noted)")

    # retrofill the unlogged 228 split (children 2306;2307 exist in gateA
    # and buildings_clean.laz; verdict in split_verdicts.csv but the
    # applying run did not land in split_applied.csv)
    ap = OREG / "split_applied.csv"
    lines = ap.read_text().strip().splitlines()
    if not any(line.startswith("228,") for line in lines):
        n06 = sum(1 for r in gate if r["instance_id"] == "2306")
        assert n06 == 1
        pts = {r["instance_id"]: r["n_points"] for r in gate
               if r["instance_id"] in ("2306", "2307")}
        lines.append(f"228,unlogged_retrofill,2306;2307,"
                     f"{pts['2306']};{pts['2307']}")
        ap.write_text("\n".join(lines) + "\n")
        print("split_applied.csv: retrofilled missing 228 row "
              "(ground_frac=unlogged_retrofill)")
    else:
        print("split_applied.csv: 228 row already present")


if __name__ == "__main__":
    if "--fix" in sys.argv:
        fix()
    else:
        assess()
