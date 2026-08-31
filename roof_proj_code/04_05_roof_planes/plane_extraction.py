#!/usr/bin/env python
"""Roof-plane extraction by region growing — production module (v3).

v3 adds a multi-scale second pass: the main pass (K=20 normals, min 55 pts)
misses small real faces (dormers, bay-window roofs, small turns) because a
20-NN neighborhood spans beyond a ~1-2 m2 face and contaminates its normals.
The second pass re-runs growing on the leftover points only, with a smaller
neighborhood (K2=10), a lower support threshold (20 pts ~ 0.7 m2) and a
STRICTER planarity gate (rms <= 4 cm) so vegetation fragments do not sneak
back in. Recovered fragments then go through the same two-stage merge, so
same-orientation eave fringes are absorbed into their parent face while
genuine dormer faces stay separate. Finally the leftover reassignment sweep
runs multiple hops to claim eave-edge points.

API: segment(P) -> (labels, planes); planes[lab] = [normal, centroid, rms,
npts]; labels -1 = unassigned.
"""

import numpy as np
from scipy.spatial import cKDTree

K = 20
ANG_GROW = 15.0
DIST_GROW = 0.15
MIN_PTS = 55
CURV_SEED = 0.08
ANG_MERGE = 8.0        # big-big merge: conservative angle
OFF_MERGE = 0.10       # big-big merge: step protection
ANG_ABSORB = 20.0      # small-into-large absorption
OFF_ABSORB = 0.12
SMALL_PTS = 150
LARGE_PTS = 300
RMS_GUARD_ABS = 0.05
RMS_GUARD_REL = 1.4
WALL_NZ = 0.34
SMOOTH_NORMALS = False

MULTISCALE = True
K2 = 10                # smaller neighborhood -> clean normals on small faces
MIN_PTS2 = 20          # ~0.7 m2 at 30 pts/m2
CURV_SEED2 = 0.12
RMS_SMALL_MAX = 0.04   # strict planarity gate for recovered small faces
REASSIGN_SWEEPS = 3

# pass 3: normal-free rescue of narrow protruding faces. Any k-NN normal is
# contaminated on a <1 m wide strip, so pass 3 clusters the remaining
# leftovers by spatial connectivity and fits each compact chunk directly
# (trimmed PCA; mini-RANSAC fallback for multi-face chunks). Floor:
# 12 pts ~ 0.4 m2 -- below that a fit is numerically meaningless at 30/m2.
PASS3 = True
MIN_PTS3 = 12
CC_LINK = 0.7          # m, leftover connectivity distance
RMS3 = 0.035
TRIM_FRAC = 0.20
MAX_DIAM3 = 8.0        # m, spatial diameter cap for a rescued mini-face


def _normals_curvature(P, nbrs):
    nb = P[nbrs]
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / nb.shape[1]
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]
    nrm[nrm[:, 2] < 0] *= -1
    curv = w[:, 0] / np.maximum(w.sum(axis=1), 1e-12)
    if SMOOTH_NORMALS:
        sm = nrm[nbrs].mean(axis=1) + nrm
        sm /= np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-12)
        sm[sm[:, 2] < 0] *= -1
        nrm = sm
    return nrm, curv


def _fit(pts):
    c = pts.mean(axis=0)
    q = pts - c
    w, v = np.linalg.eigh(q.T @ q / len(pts))
    n = v[:, 0]
    if n[2] < 0:
        n = -n
    return n, c, float(np.sqrt(max(w[0], 0.0)))


def _fit_from_moments(n, s, M):
    c = s / n
    cov = M / n - np.outer(c, c)
    w, v = np.linalg.eigh(cov)
    nv = v[:, 0]
    if nv[2] < 0:
        nv = -nv
    return nv, c, float(np.sqrt(max(w[0], 0.0)))


def _grow(P, nbrs, nrm, curv, min_pts, curv_seed, dists=None, max_edge=1.0):
    """seeded region growing; returns labels (-1 unassigned) and label count"""
    n = len(P)
    cos_grow = np.cos(np.radians(ANG_GROW))
    labels = np.full(n, -1, np.int32)
    next_lab = 0
    for seed in np.argsort(curv):
        if labels[seed] != -1 or curv[seed] > curv_seed:
            continue
        rn = nrm[seed].copy(); rc = P[seed].copy()
        region = [seed]
        labels[seed] = -2
        queue = [seed]
        since = 0
        while queue:
            i = queue.pop()
            for jj, j in enumerate(nbrs[i]):
                if labels[j] != -1:
                    continue
                if dists is not None and dists[i, jj] > max_edge:
                    continue
                if abs(np.dot(nrm[j], rn)) < cos_grow:
                    continue
                if abs(np.dot(P[j] - rc, rn)) > DIST_GROW:
                    continue
                labels[j] = -2
                region.append(j)
                queue.append(j)
                since += 1
                if since >= 256:
                    rn, rc, _ = _fit(P[region])
                    since = 0
        if len(region) >= min_pts:
            labels[np.array(region)] = next_lab
            next_lab += 1
        else:
            labels[np.array(region)] = -1
    return labels, next_lab


def _merge(P, labels, nbrs, planes, mom):
    """two-stage rms-guarded merging, in place, to convergence"""
    cos_merge = np.cos(np.radians(ANG_MERGE))
    cos_absorb = np.cos(np.radians(ANG_ABSORB))
    for _ in range(8):
        lab_n = labels[nbrs]
        mine = np.broadcast_to(labels[:, None], lab_n.shape)
        mask = (lab_n >= 0) & (mine >= 0) & (lab_n != mine)
        pairs = {(min(a, b), max(a, b))
                 for a, b in zip(mine[mask].tolist(), lab_n[mask].tolist())}
        remap = {l: l for l in planes}
        merged_any = False
        for a, b in sorted(pairs):
            ra, rb = remap[a], remap[b]
            if ra == rb:
                continue
            n1, c1, r1, k1 = planes[ra]
            n2, c2, r2, k2 = planes[rb]
            small, large = (k1, k2) if k1 <= k2 else (k2, k1)
            absorb = small < SMALL_PTS and large >= LARGE_PTS
            cos_thr = cos_absorb if absorb else cos_merge
            off_thr = OFF_ABSORB if absorb else OFF_MERGE
            if abs(np.dot(n1, n2)) < cos_thr:
                continue
            if abs(np.dot(c2 - c1, n1)) > off_thr or abs(np.dot(c1 - c2, n2)) > off_thr:
                continue
            nm = mom[ra][0] + mom[rb][0]
            sm = mom[ra][1] + mom[rb][1]
            Mm = mom[ra][2] + mom[rb][2]
            nv, c, rms = _fit_from_moments(nm, sm, Mm)
            r_ref = max(r1, r2) if not absorb else (r1 if k1 >= k2 else r2)
            if rms > max(RMS_GUARD_ABS, RMS_GUARD_REL * r_ref):
                continue
            tgt, src = (ra, rb) if k1 >= k2 else (rb, ra)
            labels[labels == src] = tgt
            mom[tgt] = [nm, sm, Mm]
            planes[tgt] = [nv, c, rms, nm]
            del planes[src], mom[src]
            for kk, vv in remap.items():
                if vv == src:
                    remap[kk] = tgt
            merged_any = True
        if not merged_any:
            break


def _coherent(pts, mask):
    """Refine a candidate mini-plane: keep only the largest spatially
    connected component of its inliers, enforce size + diameter caps,
    refit. Returns (mask, normal, centroid, rms) or None."""
    sub_idx = np.nonzero(mask)[0]
    sub = pts[sub_idx]
    tq = cKDTree(sub)
    prs = tq.query_pairs(CC_LINK, output_type="ndarray")
    parent = np.arange(len(sub))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in prs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(len(sub))])
    vals, cnts = np.unique(roots, return_counts=True)
    big = vals[int(np.argmax(cnts))]
    sel = roots == big
    if sel.sum() < MIN_PTS3:
        return None
    kept = sub[sel]
    ext = kept.max(axis=0) - kept.min(axis=0)
    if float(np.hypot(ext[0], ext[1])) > MAX_DIAM3:
        return None
    nv, c, rms = _fit(kept)
    if rms > RMS3:
        return None
    out_mask = np.zeros(len(pts), bool)
    out_mask[sub_idx[sel]] = True
    return out_mask, nv, c, rms


def _fit_chunk(pts):
    """Robust plane extraction from one compact leftover chunk.
    Returns a list of (mask over pts, normal, centroid, rms)."""
    out = []
    nv, c, rms = _fit(pts)                       # 1) whole-chunk fit
    if rms <= RMS3:
        r = _coherent(pts, np.ones(len(pts), bool))
        return [r] if r else []
    rr = np.abs((pts - c) @ nv)                  # 1b) trimmed refit
    keep = rr <= np.quantile(rr, 1 - TRIM_FRAC)
    if keep.sum() >= MIN_PTS3:
        nv2, c2, rms2 = _fit(pts[keep])
        if rms2 <= RMS3:
            inl = np.abs((pts - c2) @ nv2) <= max(2.5 * rms2, 0.03)
            if inl.sum() >= MIN_PTS3:
                r = _coherent(pts, inl)
                if r:
                    return [r]
    if len(pts) >= 2 * MIN_PTS3:                 # 2) mini-RANSAC fallback
        rng = np.random.default_rng(12345)
        remaining = np.arange(len(pts))
        for _ in range(3):
            if len(remaining) < MIN_PTS3:
                break
            R = pts[remaining]
            idx = rng.integers(0, len(R), size=(200, 3))
            p0, p1, p2 = R[idx[:, 0]], R[idx[:, 1]], R[idx[:, 2]]
            nvs = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(nvs, axis=1)
            ok = nn > 1e-9
            if not ok.any():
                break
            nvs = nvs[ok] / nn[ok, None]
            d0 = -(nvs * p0[ok]).sum(1)
            dist = np.abs(R @ nvs.T + d0)
            cnt = (dist <= RMS3).sum(axis=0)
            j = int(np.argmax(cnt))
            if cnt[j] < MIN_PTS3:
                break
            inl = dist[:, j] <= RMS3
            nv2, c2, rms2 = _fit(R[inl])
            if rms2 > RMS3:
                break
            inl = np.abs((R - c2) @ nv2) <= max(2.5 * rms2, 0.03)
            if inl.sum() < MIN_PTS3:
                break
            gmask = np.zeros(len(pts), bool)
            gmask[remaining[inl]] = True
            r = _coherent(pts, gmask)
            if r is None:
                remaining = remaining[~inl]
                continue
            out.append(r)
            remaining = remaining[~r[0][remaining]]
    return out


def segment(P):
    n = len(P)
    tree = cKDTree(P)
    dists, nbrs = tree.query(P, k=K + 1)
    nbrs = nbrs[:, 1:]; dists = dists[:, 1:]
    nrm, curv = _normals_curvature(P, nbrs)
    labels, next_lab = _grow(P, nbrs, nrm, curv, MIN_PTS, CURV_SEED, dists)

    mom, planes = {}, {}
    for lab in range(next_lab):
        pts = P[labels == lab]
        mom[lab] = [len(pts), pts.sum(axis=0), pts.T @ pts]
        nv, c, rms = _fit_from_moments(*mom[lab])
        planes[lab] = [nv, c, rms, len(pts)]
    _merge(P, labels, nbrs, planes, mom)

    # ---- multi-scale second pass on leftovers (small real faces) ----
    if MULTISCALE:
        un_idx = np.nonzero(labels < 0)[0]
        if len(un_idx) >= max(MIN_PTS2, K2 + 2):
            Q = P[un_idx]
            tree2 = cKDTree(Q)
            d2, nbrs2 = tree2.query(Q, k=K2 + 1)
            nbrs2 = nbrs2[:, 1:]; d2 = d2[:, 1:]
            nrm2, curv2 = _normals_curvature(Q, nbrs2)
            lab2, nl2 = _grow(Q, nbrs2, nrm2, curv2, MIN_PTS2, CURV_SEED2, d2)
            added = False
            for l2 in range(nl2):
                sel = lab2 == l2
                pts = Q[sel]
                nv, c, rms = _fit(pts)
                if rms > RMS_SMALL_MAX:
                    continue
                gl = max(planes.keys(), default=-1) + 1
                labels[un_idx[sel]] = gl
                mom[gl] = [len(pts), pts.sum(axis=0), pts.T @ pts]
                planes[gl] = [nv, c, rms, len(pts)]
                added = True
            if added:
                _merge(P, labels, nbrs, planes, mom)

    # ---- pass 3: normal-free rescue of narrow protruding faces ----
    if PASS3:
        un_idx = np.nonzero(labels < 0)[0]
        if len(un_idx) >= MIN_PTS3:
            Q = P[un_idx]
            tq = cKDTree(Q)
            prs = tq.query_pairs(CC_LINK, output_type="ndarray")
            parent = np.arange(len(Q))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for a, b in prs:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            roots = np.array([find(i) for i in range(len(Q))])
            added = False
            for r in np.unique(roots):
                cc = np.nonzero(roots == r)[0]
                if len(cc) < MIN_PTS3:
                    continue
                for mask, nv, c, rms in _fit_chunk(Q[cc]):
                    gl = max(planes.keys(), default=-1) + 1
                    sel = un_idx[cc[mask]]
                    labels[sel] = gl
                    pts = P[sel]
                    mom[gl] = [len(pts), pts.sum(axis=0), pts.T @ pts]
                    planes[gl] = [nv, c, rms, len(pts)]
                    added = True
            if added:
                _merge(P, labels, nbrs, planes, mom)

    # ---- multi-hop leftover reassignment (eave fringes) ----
    cos_assign = np.cos(np.radians(25.0))
    for _ in range(REASSIGN_SWEEPS):
        changed = False
        for i in np.nonzero(labels < 0)[0]:
            for j in nbrs[i]:
                lab = labels[j]
                if lab < 0:
                    continue
                nv, c, _, _ = planes[lab]
                if abs(np.dot(P[i] - c, nv)) <= DIST_GROW and \
                        abs(np.dot(nrm[i], nv)) >= cos_assign:
                    labels[i] = lab
                    changed = True
                    break
        if not changed:
            break

    for lab in list(planes):
        k = int((labels == lab).sum())
        if k == 0:
            del planes[lab]
        else:
            planes[lab][3] = k
    return labels, planes
