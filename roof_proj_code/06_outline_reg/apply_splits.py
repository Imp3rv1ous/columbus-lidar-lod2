#!/usr/bin/env python
"""Apply TWO_BUILDINGS verdicts: split under-segmented instances in
buildings_clean.laz, then the downstream chain is rerun externally.

For each candidate: rebuild the 0.5 m closed mask from ALL its points,
take components >=25% of the largest (falling back to the opened-base
recipe when closing bridges the gap), re-run the ground-in-gap test, and
if TWO_BUILDINGS: relabel points per component (leftovers -> nearest big
component). New ids continue after the current max. gateA table updated:
parent row marked split_parent, child rows appended."""

import csv
import time
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code")
ISEG = BASE / "02_03_instance_seg/output_v2"
PRED = BASE / "01_semantic_randlanet/results/preds"
OUT = BASE / "06_outline_reg/output"
FT_TO_M = 0.30480060960121924
CELL = 0.5
GCELL = 1.0
SPLIT_FRAC = 0.35
CANDIDATES = [209, 217, 228, 1244, 1703, 2098, 216]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    # DTM presence grid
    tiles = sorted(PRED.glob("*_pred.laz"))
    gmin = np.array([np.inf, np.inf])
    for t in tiles:
        with laspy.open(t) as r:
            gmin = np.minimum(gmin, r.header.mins[:2])
    x0g, y0g = gmin * FT_TO_M
    has_parts = []
    for t in tiles:
        l2 = laspy.read(t)
        m = np.asarray(l2.classification) == 2
        has_parts.append((np.asarray(l2.x)[m] * FT_TO_M,
                          np.asarray(l2.y)[m] * FT_TO_M))
    gx = np.concatenate([p[0] for p in has_parts])
    gy = np.concatenate([p[1] for p in has_parts])
    nxg = int((gx.max() - x0g) / GCELL) + 2
    nyg = int((gy.max() - y0g) / GCELL) + 2
    has_g = np.zeros((nyg, nxg), bool)
    has_g[((gy - y0g) / GCELL).astype(np.int32),
          ((gx - x0g) / GCELL).astype(np.int32)] = True

    cl = laspy.read(ISEG / "buildings_clean.laz")
    inst = np.asarray(cl.instance_id).copy()
    X = np.column_stack([np.asarray(cl.x), np.asarray(cl.y)]) * FT_TO_M
    next_id = int(inst.max()) + 1
    s8 = np.ones((3, 3), bool)
    mapping = []
    for fid in CANDIDATES:
        sel = np.nonzero(inst == fid)[0]
        if len(sel) == 0:
            continue
        P = X[sel]
        x0m, y0m = P[:, 0].min(), P[:, 1].min()
        gxc = ((P[:, 0] - x0m) / CELL).astype(np.int32)
        gyc = ((P[:, 1] - y0m) / CELL).astype(np.int32)
        mm = np.zeros((gyc.max() + 3, gxc.max() + 3), np.uint16)
        np.add.at(mm, (gyc + 1, gxc + 1), 1)
        for recipe in ("closed", "opened"):
            if recipe == "closed":
                base = ndimage.binary_fill_holes(
                    ndimage.binary_closing(mm >= 1, structure=s8))
            else:
                base = ndimage.binary_opening(mm >= 1, structure=s8)
            lab, n = ndimage.label(base, structure=s8)
            if n < 2:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            big = [i + 1 for i, s in enumerate(sizes) if s >= 0.25 * sizes.max()]
            if len(big) >= 2:
                break
        else:
            log(f"id {fid}: stays connected under both recipes -> keep")
            continue
        A = lab == big[0]; Bm = lab == big[1]
        dA = ndimage.distance_transform_edt(~A) * CELL
        dB = ndimage.distance_transform_edt(~Bm) * CELL
        corridor = (dA <= 3.0) & (dB <= 3.0) & ~A & ~Bm
        cy, cx = np.nonzero(corridor)
        if len(cy) < 4:
            log(f"id {fid}: no corridor -> keep")
            continue
        wx = (cx - 1 + 0.5) * CELL + x0m
        wy = (cy - 1 + 0.5) * CELL + y0m
        frac = float(has_g[np.clip(((wy - y0g) / GCELL).astype(np.int32), 0, nyg - 1),
                           np.clip(((wx - x0g) / GCELL).astype(np.int32), 0, nxg - 1)].mean())
        if frac < SPLIT_FRAC:
            log(f"id {fid}: ground-in-gap {frac:.2f} -> occluded single, keep")
            continue
        # assign every point to nearest big component
        big_mask = np.isin(lab, big)
        lab_big = np.where(big_mask, lab, 0)
        _, (iyn, ixn) = ndimage.distance_transform_edt(lab_big == 0,
                                                       return_indices=True)
        lab_fill = lab_big[iyn, ixn]
        part = lab_fill[gyc + 1, gxc + 1]
        new_ids = {}
        for j, comp in enumerate(big):
            nid = next_id + j
            new_ids[comp] = nid
            inst[sel[part == comp]] = nid
        next_id += len(big)
        mapping.append((fid, frac, {int(k): int(v) for k, v in new_ids.items()},
                        [int((part == c).sum()) for c in big]))
        log(f"id {fid}: SPLIT (ground {frac:.2f}) -> "
            f"{[v for v in new_ids.values()]} with pts {[int((part==c).sum()) for c in big]}")

    cl.instance_id = inst
    cl.write(ISEG / "buildings_clean.laz")
    log(f"wrote buildings_clean.laz: {len(np.unique(inst[inst>0]))} instances")

    rows = list(csv.DictReader(open(ISEG / "instances_gateA.csv")))
    by_id = {int(r["instance_id"]): r for r in rows}
    add = []
    for fid, frac, ids, npts in mapping:
        parent = by_id.get(fid)
        if parent:
            parent["decision"] = f"split_parent({','.join(str(v) for v in ids.values())})"
        for (comp, nid), np_ in zip(ids.items(), npts):
            r = dict(parent) if parent else {k: "" for k in rows[0]}
            r["instance_id"] = str(nid)
            r["n_points"] = str(np_)
            r["decision"] = f"keep_split_child_of_{fid}"
            add.append(r)
    rows.extend(add)
    with open(ISEG / "instances_gateA.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    with open(OUT / "split_applied.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent_id", "ground_frac", "child_ids", "child_pts"])
        for fid, frac, ids, npts in mapping:
            w.writerow([fid, frac, ";".join(str(v) for v in ids.values()),
                        ";".join(str(n) for n in npts)])
    log(f"applied {len(mapping)} splits")


if __name__ == "__main__":
    main()
