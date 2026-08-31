"""Rule-based DRAFT segmentation for hand-correction (predict-then-correct).

Produces a CloudCompare-ready `<tile>_draft.laz` whose `classification` field is a
DRAFT label (ASPRS 2/5/6/1, 0=left-unlabeled) so the user opens it, colours by
classification, and only FIXES mistakes instead of labelling from scratch.

Method (same feature recipe as training; three orthogonal cuts):
  ground(2)      USGS classification==2 (trusted as-is)
  vegetation(5)  non-ground & num_returns>=2 & HAG>VEG_HAG          (multi-return = canopy)
  building(6)    non-ground & single-return & HAG>BLD_HAG &
                 part of a large, compact, connected planar blob     (roof)
  others(1)      rejected tall thin/small clusters (wires, poles)
  unlabeled(0)   near-ground ambiguous (thin grass / missed ground)  -> left for the user

Never overwrites the original tile; writes a NEW file next to it (and an `hag_m`
extra dimension so CloudCompare gets a clean height slider too).

  python3 scripts/draft_segment.py BS838711
  python3 scripts/draft_segment.py BS838711 BS842712 BS841710 BS843711
"""
from __future__ import annotations

import sys
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config, FT2M
from randla_net.prep.hag import compute_hag

# --- tunable thresholds (METERS) -------------------------------------------------
VEG_HAG = 0.5      # below this, near-ground multi-return is more likely low grass -> leave
BLD_HAG = 2.5      # roofs are well above this; cars/fences are below
GRASS_HAG = 0.5    # non-ground single-return below this = ambiguous grass -> leave unlabeled
CELL = 1.0         # building connected-component grid cell (m)
MIN_CELLS = 15     # min footprint area (cells ~= m^2) for a blob to count as a building
MIN_FILL = 0.30    # min (cells / bbox-area): drops thin lines (power wires, road edges)
ROOF_MARGIN = 1.0  # in-footprint points up to roof+this are building; above = overhanging tree
MR_FRAC_MAX = 0.50 # a footprint is dense veg (not a roof) only if it is BOTH this multi-return ...
ROUGH_MAX = 2.5    # ... AND this vertically rough (HAG std, m). Roofs are planar -> protected.
OTHERS_HAG_LO = 0.5
OTHERS_HAG_HI = 2.5
ISO_RADIUS = 2.0   # despeckle: a point needs ISO_MIN_SAME same-class neighbours within this
ISO_MIN_SAME = 6   # else it is isolated salt-and-pepper -> demoted to unlabeled (0, ignored)
CONSIST_K = 12     # neighbourhood-consistency: look at this many nearest labeled non-ground nbrs
CONSIST_MIN = 0.35 # if < this fraction share the point's class, it is a boundary minority -> demote
VEG_GROW_R = 2.0   # an nr==2 point is veg only if a confident (nr>=3) canopy point is within this;
                   # else it's a building-edge / pole / isolated nr==2 -> left unlabeled (not veg)

# internal code -> ASPRS (0 stays 0 = "never classified" = left for the user)
_LUT = np.array([0, 1, 2, 5, 6], dtype=np.uint8)  # idx = internal 0..4
_NAMES = {0: "unlabeled", 1: "others", 2: "ground", 3: "vegetation", 4: "building"}


def building_footprint(xyz_m: np.ndarray, core: np.ndarray):
    """Grid the tall single-return cores -> keep large compact connected blobs.

    Returns (origin_ij, H, W, kept_grid) where kept_grid is an (H,W) int array of
    cluster ids (1..K for kept buildings, 0 elsewhere)."""
    idx = np.where(core)[0]
    if idx.size == 0:
        return np.zeros(2, np.int64), 1, 1, np.zeros((1, 1), np.int64)
    ij = np.floor(xyz_m[idx, :2] / CELL).astype(np.int64)
    origin_ij = ij.min(axis=0)
    ij -= origin_ij
    H, W = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    grid = np.zeros((H, W), dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, structure=np.ones((3, 3), bool))  # fill 1-cell roof gaps
    lab, n = ndimage.label(grid, structure=np.ones((3, 3), bool))         # 8-connectivity
    remap = np.zeros(n + 1, dtype=np.int64)
    objs = ndimage.find_objects(lab)
    nxt = 1
    for c in range(1, n + 1):
        sl = objs[c - 1]
        cells = int((lab[sl] == c).sum())
        if cells < MIN_CELLS:
            continue
        bbox_area = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        if cells / bbox_area < MIN_FILL:          # thin/linear -> wire, not building
            continue
        remap[c] = nxt
        nxt += 1
    return origin_ij, H, W, remap[lab]


def point_clusters(xyz_m, mask, origin_ij, H, W, kept_grid) -> np.ndarray:
    """Footprint cluster id (0 = outside any building) for every point in `mask`."""
    out = np.zeros(mask.shape[0], dtype=np.int64)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return out
    ij = np.floor(xyz_m[idx, :2] / CELL).astype(np.int64) - origin_ij
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < H) & (ij[:, 1] >= 0) & (ij[:, 1] < W)
    cl = np.zeros(idx.size, dtype=np.int64)
    cl[inb] = kept_grid[ij[inb, 0], ij[inb, 1]]
    out[idx] = cl
    return out


def near_points(xyz_m, cand, core, radius) -> np.ndarray:
    """Bool mask of `cand` points that have at least one `core` point within `radius`."""
    from scipy.spatial import cKDTree
    out = np.zeros(cand.shape, dtype=bool)
    ci = np.where(cand)[0]
    ki = np.where(core)[0]
    if ci.size == 0 or ki.size == 0:
        return out
    tree = cKDTree(xyz_m[ki])
    d, _ = tree.query(xyz_m[ci], k=1, workers=-1)
    out[ci] = d <= radius
    return out


def compact_flat_clusters(xyz_m, hag, cand) -> np.ndarray:
    """Point mask of `cand` points that fall in large, compact, FLAT grid blobs (roof-like).
    Same size+fill filters as the footprint detector, plus a per-cluster HAG-std (flatness)
    gate so dense canopy clumps are excluded."""
    out = np.zeros(cand.shape, dtype=bool)
    idx = np.where(cand)[0]
    if idx.size == 0:
        return out
    ij = np.floor(xyz_m[idx, :2] / CELL).astype(np.int64)
    ij -= ij.min(axis=0)
    H, W = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    grid = np.zeros((H, W), dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, structure=np.ones((3, 3), bool))
    lab, n = ndimage.label(grid, structure=np.ones((3, 3), bool))
    if n == 0:
        return out
    cid = lab[ij[:, 0], ij[:, 1]]
    keep = np.zeros(n + 1, dtype=bool)
    objs = ndimage.find_objects(lab)
    for c in range(1, n + 1):
        sl = objs[c - 1]
        cells = int((lab[sl] == c).sum())
        if cells < MIN_CELLS:
            continue
        bbox = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        if cells / bbox < MIN_FILL:                # thin/linear -> tree row, not roof
            continue
        if hag[idx[cid == c]].std() > ROUGH_MAX:   # rough -> canopy, not roof
            continue
        keep[c] = True
    out[idx[keep[cid]]] = True
    return out


def segment_tile(xyz_m, cls, rn, nr, hag) -> np.ndarray:
    """Return internal labels 0..4 for every point (footprint-first, then height-split)."""
    lab = np.zeros(cls.shape[0], dtype=np.int64)
    ground = cls == 2
    nonground = ~ground
    lab[ground] = 2

    # 1) building footprints from clean tall single-return cores
    core = nonground & (nr <= 1) & (hag > BLD_HAG)
    origin_ij, H, W, kept_grid = building_footprint(xyz_m, core)
    K = int(kept_grid.max())

    ng_cl = point_clusters(xyz_m, nonground, origin_ij, H, W, kept_grid)  # cluster id per nonground pt
    in_fp = ng_cl > 0

    # reject a footprint as dense vegetation only if it is BOTH multi-return-heavy AND vertically
    # rough; a planar roof (low HAG std) survives even with some overhanging-tree multi-returns.
    mr = (nr >= 2).astype(np.float64)
    tot = np.bincount(ng_cl, minlength=K + 1).astype(np.float64)
    mrc = np.bincount(ng_cl, weights=mr, minlength=K + 1)
    frac = mrc / np.maximum(tot, 1.0)
    sum_h = np.bincount(ng_cl, weights=hag, minlength=K + 1)
    sum_h2 = np.bincount(ng_cl, weights=hag * hag, minlength=K + 1)
    mean_h = sum_h / np.maximum(tot, 1.0)
    std_h = np.sqrt(np.maximum(sum_h2 / np.maximum(tot, 1.0) - mean_h ** 2, 0.0))
    is_veg_cluster = (frac > MR_FRAC_MAX) & (std_h > ROUGH_MAX)
    is_bldg_cluster = np.zeros(K + 1, dtype=bool)
    is_bldg_cluster[1:] = ~is_veg_cluster[1:]
    bldg_pt = is_bldg_cluster[ng_cl]                       # point sits in a kept BUILDING footprint
    veg_cluster_pt = in_fp & (~bldg_pt)                    # point sits in a rejected (veg) footprint

    # per-building roof height = 90th pct of core HAG (building clusters only)
    roof_h = np.zeros(K + 1, dtype=np.float64)
    core_cl = np.where(core, ng_cl, 0)
    for c in range(1, K + 1):
        if not is_bldg_cluster[c]:
            continue
        hs = hag[core_cl == c]
        if hs.size:
            roof_h[c] = np.percentile(hs, 90)
    rh_pt = roof_h[ng_cl]                                   # local roof height under each point (0 outside)

    # 2) inside a BUILDING footprint, at/below roof+margin -> building (reclaims multi-return roof edges)
    building = nonground & bldg_pt & (hag > GRASS_HAG) & (hag <= rh_pt + ROOF_MARGIN)
    lab[building] = 4

    # 3) vegetation. In the open (outside any footprint), nr>=3 is confident canopy; an nr==2
    #    point counts as canopy only if a confident canopy point is within VEG_GROW_R (else it is
    #    a building-edge ring / pole / isolated nr==2 and is left unlabeled, NOT forced into veg).
    canopy_core = nonground & (~building) & (~in_fp) & (nr >= 3) & (hag > VEG_HAG)
    edge_cand = nonground & (~building) & (~in_fp) & (nr == 2) & (hag > VEG_HAG)
    canopy_edge = near_points(xyz_m, edge_cand, canopy_core, VEG_GROW_R)
    veg = nonground & (~building) & (
        (veg_cluster_pt & (hag > VEG_HAG))
        | (bldg_pt & (hag > rh_pt + ROOF_MARGIN))
        | canopy_core | canopy_edge
    )
    lab[veg] = 3

    # 4) others: leftover compact mid-height single-return objects (cars, low clutter, wires/poles)
    rest = nonground & (lab == 0)
    others = rest & (nr <= 1) & (hag > OTHERS_HAG_LO) & (hag <= OTHERS_HAG_HI)
    lab[others] = 1

    # 5) rescue: compact FLAT single-return clusters still sitting in veg are missed roofs
    #    (footprints the MR+roughness gate over-rejected). Shape gate spares tree rows/speckle.
    rescued = compact_flat_clusters(xyz_m, hag, (lab == 3) & (nr <= 1))
    lab[rescued] = 4

    # everything still 0 (near-ground ambiguous grass / missed ground) stays unlabeled
    return lab


def despeckle(xyz_m, lab, classes=(1, 3, 4)) -> dict:
    """Demote isolated points (too few same-class neighbours) to unlabeled (0).

    Uses the same-class k-NN distance: if the ISO_MIN_SAME-th nearest neighbour OF
    THE SAME CLASS is farther than ISO_RADIUS, the point is salt-and-pepper speckle
    and is set to 0 (ignored by the weakly-supervised loss). Safe by construction:
    we never invent a label, only retract a low-confidence one."""
    from scipy.spatial import cKDTree
    demoted = {}
    for c in classes:
        m = lab == c
        npts = int(m.sum())
        if npts <= ISO_MIN_SAME:
            continue
        pts = xyz_m[m]
        tree = cKDTree(pts)
        dist, _ = tree.query(pts, k=ISO_MIN_SAME + 1)   # incl. self at dist 0
        iso = dist[:, -1] > ISO_RADIUS
        idx = np.where(m)[0][iso]
        lab[idx] = 0
        demoted[c] = int(iso.sum())
    return demoted


def consistency_filter(xyz_m, lab, classes=(1, 3, 4)) -> int:
    """Demote boundary 'minority' points: a labeled point whose nearest labeled neighbours
    mostly belong to ANOTHER class is set to unlabeled (0). Cleans tree<->building mixing
    in both directions without inventing labels (we only retract)."""
    from scipy.spatial import cKDTree
    pool = np.isin(lab, classes)
    idx = np.where(pool)[0]
    if idx.size <= CONSIST_K:
        return 0
    pts = xyz_m[idx]
    labs = lab[idx]
    tree = cKDTree(pts)
    _, nbr = tree.query(pts, k=CONSIST_K + 1, workers=-1)   # incl self
    same = (labs[nbr[:, 1:]] == labs[:, None]).mean(axis=1)
    demote = same < CONSIST_MIN
    lab[idx[demote]] = 0
    return int(demote.sum())


def run(tile: str, cfg: Config) -> None:
    src = cfg.orig_laz(tile)
    las = laspy.read(str(src))
    xyz_ft = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    xyz_m = xyz_ft.astype(np.float64) * FT2M
    cls = np.asarray(las.classification).astype(np.int64)
    rn = np.asarray(las.return_number).astype(np.int64)
    nr = np.asarray(las.number_of_returns).astype(np.int64)

    hag = compute_hag(xyz_m, cls, cell=cfg.hag_cell, k=cfg.hag_knn)
    lab = segment_tile(xyz_m, cls, rn, nr, hag)
    demoted = despeckle(xyz_m, lab)
    nconsist = consistency_filter(xyz_m, lab)

    n = lab.shape[0]
    if demoted:
        print("  despeckle -> unlabeled: " +
              ", ".join(f"{_NAMES[c]} {v:,}" for c, v in demoted.items()))
    print(f"  consistency -> unlabeled: {nconsist:,}")
    print(f"\n=== {tile}  ({n:,} pts) ===")
    for c in (0, 1, 2, 3, 4):
        m = int((lab == c).sum())
        print(f"  {c} {_NAMES[c]:<11} {m:>10,}  {100*m/n:5.1f}%")

    las.classification = _LUT[lab]
    if "hag_m" not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="hag_m", type=np.float32))
    las.hag_m = hag.astype(np.float32)

    out = cfg.train_dir / tile / f"{tile}_draft.laz"
    out.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out))
    print(f"  -> {out}")


def main() -> None:
    tiles = sys.argv[1:]
    if not tiles:
        print("usage: draft_segment.py <tile> [<tile> ...]")
        sys.exit(1)
    cfg = Config()
    print(f"train_dir={cfg.train_dir}")
    for t in tiles:
        run(t, cfg)


if __name__ == "__main__":
    main()
