#!/usr/bin/env python
"""Draft surfaces: turn every extracted roof/wall plane into an actual
bounded planar facet (the user-facing proof that 'plane point clouds'
have truly become planes).

Per plane: project its points onto the fitted plane (strictly coplanar),
triangulate in the plane's 2D basis (Delaunay), drop triangles with edges
> EDGE_MAX (so facets follow the point support and don't bridge holes),
color by the same collision-free per-plane palette. All buildings go into
one binary PLY (ftUS coords -> overlays the LAZ and the cue OBJs in
CloudCompare).

This is a DRAFT preview of stage 7: edges are still jagged (regularization
is stage 6) and faces meet with small seams (snapping/assembly is stage 7).
Its purpose: (1) visible point->surface conversion; (2) the seams between
facets coincide with the stage-5 ridge/valley lines, validating the cues.

Also renders a 3-building 3D figure (surfaces + cue lines) for quick review.
"""

import colorsys
import csv
import time
from pathlib import Path

import numpy as np
import laspy
from scipy.spatial import Delaunay

BASE = Path("/Users/baioqile/Documents/roof_proj/roof_proj_code/04_05_roof_planes")
FULL = BASE / "output_full"
FT_TO_M = 0.30480060960121924
EDGE_MAX = 1.2   # m, max triangle edge (prevents bridging holes/concavities)
DEMO_IDS = [1206, 307, 34]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def palette(nmax):
    sats = [0.95, 0.7, 0.55]; vals = [0.95, 0.75, 0.6, 0.85]
    pal = np.zeros((nmax + 1, 3), np.uint8); pal[0] = (200, 200, 200)
    phi = 0.6180339887498949; seen = set()
    for i in range(1, nmax + 1):
        h = (i * phi) % 1.0; s = sats[i % 3]; v = vals[i % 4]
        c = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
        while c in seen:
            v = v - 0.03 if v > 0.2 else v + 0.35
            c = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, min(v, 1.0)))
        seen.add(c); pal[i] = c
    return pal


def main():
    las = laspy.read(FULL / "buildings_planes.laz")
    pid = np.asarray(las.plane_id)
    styp = np.asarray(las.surf_type)
    inst = np.asarray(las.instance_id)
    X = np.column_stack([np.asarray(las.x), np.asarray(las.y),
                         np.asarray(las.z)]) * FT_TO_M
    pl = {}
    with open(FULL / "planes.csv") as f:
        for r in csv.DictReader(f):
            pl[int(r["plane_id"])] = (
                np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])]),
                np.array([float(r["cx_m"]), float(r["cy_m"]), float(r["cz_m"])]))
    nmax = int(pid.max())
    pal = palette(nmax)

    order = np.argsort(pid, kind="stable")
    u, s = np.unique(pid[order], return_index=True)
    e = np.append(s[1:], len(order))

    all_v, all_c, all_f = [], [], []
    voff = 0
    demo = {i: {"tri": [], "col": []} for i in DEMO_IDS}
    t0 = time.time()
    nplanes = 0
    for g, a, b in zip(u, s, e):
        if g == 0:
            continue
        idx = order[a:b]
        P = X[idx]
        n, c = pl[int(g)]
        # strict projection onto the fitted plane
        d = (P - c) @ n
        Q = P - d[:, None] * n[None, :]
        # 2D basis in the plane
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        uu = np.cross(n, ref); uu /= np.linalg.norm(uu)
        vv = np.cross(n, uu)
        uv = np.column_stack([(Q - c) @ uu, (Q - c) @ vv])
        if len(uv) < 3:
            continue
        try:
            tri = Delaunay(uv)
        except Exception:
            continue
        T = tri.simplices
        # edge-length filter
        e01 = np.linalg.norm(Q[T[:, 0]] - Q[T[:, 1]], axis=1)
        e12 = np.linalg.norm(Q[T[:, 1]] - Q[T[:, 2]], axis=1)
        e20 = np.linalg.norm(Q[T[:, 2]] - Q[T[:, 0]], axis=1)
        T = T[(e01 <= EDGE_MAX) & (e12 <= EDGE_MAX) & (e20 <= EDGE_MAX)]
        if len(T) == 0:
            continue
        col = pal[int(g)]
        if styp[idx[0]] == 2:
            col = (col * 0.35).astype(np.uint8)
        all_v.append(Q / FT_TO_M)          # back to ftUS for overlay
        all_c.append(np.tile(col, (len(Q), 1)))
        all_f.append(T + voff)
        voff += len(Q)
        nplanes += 1
        fid = int(inst[idx[0]])
        if fid in demo:
            demo[fid]["tri"].append(Q[T])   # (m,3,3) meters
            demo[fid]["col"].append(np.tile(col / 255.0, (len(T), 1)))

    V = np.vstack(all_v); C = np.vstack(all_c); F = np.vstack(all_f)
    log(f"{nplanes} facets, {len(V):,} vertices, {len(F):,} triangles "
        f"({time.time()-t0:.0f}s)")

    # ---- binary PLY ----
    ply = FULL / "draft_surfaces.ply"
    with open(ply, "wb") as f:
        f.write((
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(V)}\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(F)}\n"
            "property list uchar uint vertex_indices\nend_header\n").encode())
        vrec = np.zeros(len(V), dtype=[("xyz", "<f8", 3), ("rgb", "u1", 3)])
        vrec["xyz"] = V; vrec["rgb"] = C
        vrec.tofile(f)
        frec = np.zeros(len(F), dtype=[("n", "u1"), ("idx", "<u4", 3)])
        frec["n"] = 3; frec["idx"] = F
        frec.tofile(f)
    log(f"wrote {ply.name} ({ply.stat().st_size/1e6:.0f} MB)")

    # ---- demo figure: surfaces + cue lines ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    cues = {i: [] for i in DEMO_IDS}
    with open(FULL / "cues.csv") as f:
        for r in csv.DictReader(f):
            fid = int(r["instance_id"])
            if fid in cues:
                cues[fid].append((r["type"],
                                  [float(r["x1"]), float(r["y1"]), float(r["z1"])],
                                  [float(r["x2"]), float(r["y2"]), float(r["z2"])]))
    fig = plt.figure(figsize=(19, 6.5))
    for k, fid in enumerate(DEMO_IDS):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        tris = np.vstack(demo[fid]["tri"])
        cols = np.vstack(demo[fid]["col"])
        ctr = tris.reshape(-1, 3).mean(axis=0)
        tris = tris - ctr
        pc = Poly3DCollection(tris, facecolors=cols, edgecolors="none", alpha=1.0)
        ax.add_collection3d(pc)
        segs, segc = [], []
        for t, p1, p2 in cues[fid]:
            segs.append([np.array(p1) - ctr + [0, 0, 0.06],
                         np.array(p2) - ctr + [0, 0, 0.06]])
            segc.append({"ridge": "red", "valley": "magenta",
                         "step": "blue"}[t])
        if segs:
            ax.add_collection3d(Line3DCollection(segs, colors=segc, linewidths=2.5))
        ext = np.abs(tris.reshape(-1, 3)).max()
        ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
        ax.set_zlim(-ext / 2, ext / 2)
        ax.view_init(elev=38, azim=-55)
        ax.set_axis_off()
        ax.set_title(f"id {fid}: {sum(len(t) for t in demo[fid]['tri']):,} facet "
                     f"triangles + stage-5 cue lines", fontsize=10)
    fig.suptitle("Point clouds turned into planar facets — seams coincide with "
                 "ridge/valley lines (red/magenta)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FULL / "qa_draft_surfaces.png", dpi=150)
    log("wrote qa_draft_surfaces.png")


if __name__ == "__main__":
    main()
