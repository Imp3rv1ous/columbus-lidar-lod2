"""Unit tests for Phase 1/2 prep core logic, on synthetic data with known answers.

Run:  python3 tests/test_prep.py     (self-contained, no pytest needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import FeatureRecipe
from randla_net.prep.grid_subsample import grid_subsample
from randla_net.prep.hag import compute_hag
from randla_net.prep.features import build_features
from randla_net.prep.labels import assign_labels


def test_grid_subsample():
    # grid=1.0: two points in voxel(0,0,0), one in voxel(2,0,0)
    xyz = np.array([[0.2, 0.2, 0.2], [0.8, 0.4, 0.6], [2.5, 0.5, 0.5]])
    feats = np.array([[1.0], [3.0], [5.0]])
    sx, sf, inv = grid_subsample(xyz, feats, grid=1.0)
    assert sx.shape == (2, 3), sx.shape
    # voxel(0,0,0) barycenter = mean of first two points
    assert np.allclose(sx[0], [0.5, 0.3, 0.4]), sx[0]
    assert np.allclose(sf[0], [2.0]) and np.allclose(sf[1], [5.0])
    assert inv.tolist() == [0, 0, 1], inv.tolist()
    # reconstruction: every full point within its voxel diagonal of its barycenter
    d = np.linalg.norm(sx[inv] - xyz, axis=1)
    assert d.max() < 1.0 * np.sqrt(3), d.max()
    # determinism
    sx2, _, inv2 = grid_subsample(xyz, feats, grid=1.0)
    assert np.array_equal(sx, sx2) and np.array_equal(inv, inv2)


def test_hag_flat_and_sloped():
    rng = np.random.RandomState(0)
    # flat ground z=0 (class 2) + elevated points z=5 (class 1)
    gx, gy = np.meshgrid(np.arange(0, 20, 0.5), np.arange(0, 20, 0.5))
    g = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    e = np.column_stack([rng.uniform(2, 18, 200), rng.uniform(2, 18, 200), np.full(200, 5.0)])
    xyz = np.vstack([g, e])
    cls = np.array([2] * len(g) + [1] * len(e))
    hag = compute_hag(xyz, cls, cell=1.0, k=4)
    assert abs(np.median(hag[cls == 2])) < 0.05, np.median(hag[cls == 2])
    assert abs(np.median(hag[cls == 1]) - 5.0) < 0.2, np.median(hag[cls == 1])

    # sloped ground z = 0.2*x ; elevated 5 m above local ground -> HAG ~5 regardless of x
    gz = 0.2 * g[:, 0]
    gs = np.column_stack([g[:, 0], g[:, 1], gz])
    es = np.column_stack([e[:, 0], e[:, 1], 0.2 * e[:, 0] + 5.0])
    xyz2 = np.vstack([gs, es])
    cls2 = cls
    hag2 = compute_hag(xyz2, cls2, cell=1.0, k=4)
    assert abs(np.median(hag2[cls2 == 2])) < 0.1, np.median(hag2[cls2 == 2])
    assert abs(np.median(hag2[cls2 == 1]) - 5.0) < 0.3, np.median(hag2[cls2 == 1])


def test_features_recipe():
    tile = {
        "intensity": np.array([0.0, 100.0, 200.0, 300.0, 65535.0]),
        "return_number": np.array([1.0, 1.0, 2.0, 7.0, 1.0]),
        "num_returns": np.array([1.0, 1.0, 3.0, 7.0, 1.0]),
    }
    hag = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    feats = build_features(tile, hag, FeatureRecipe())
    assert feats.shape == (5, 4), feats.shape
    intn, rnn, nrn, h = feats.T
    assert intn.min() >= 0.0 and intn.max() <= 1.0
    assert np.allclose(rnn, np.clip(tile["return_number"] / 7.0, 0, 1))
    assert np.allclose(nrn, np.clip(tile["num_returns"] / 7.0, 0, 1))
    assert np.allclose(h, hag)  # hag passed through unchanged, last column
    # different recipe -> different in_dim, hag-only
    only = build_features(tile, hag, FeatureRecipe(channels=("hag",)))
    assert only.shape == (5, 1) and np.allclose(only[:, 0], hag)


def test_label_mapping():
    # 5 voxels on a line; first two are ground
    sub_xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], float)
    is_ground = np.array([True, True, False, False, False])
    annotations = [
        # building near voxel2 (claims it)
        ("building", 4, np.array([[2.05, 0, 0],
                                  [1.10, 0, 0]])),   # this one near voxel1 (ground), dist 0.10
        # veg exactly on voxel1 (dist 0 -> beats the building's 0.10 -> overrides ground)
        ("vegetation", 3, np.array([[1.00, 0, 0],
                                    [3.00, 0, 0]])),  # also voxel3
        # one annotation far from any voxel -> rejected by guard
        ("others", 1, np.array([[10.0, 0, 0]])),
    ]
    labels, stats = assign_labels(sub_xyz, is_ground, annotations, guard=0.5)
    # voxel0 stays ground; voxel1 -> veg (nearest wins, overrides ground); voxel2 -> building;
    # voxel3 -> veg; voxel4 -> unlabeled
    assert labels.tolist() == [2, 3, 4, 3, 0], labels.tolist()
    assert stats["others"]["rejected"] == 1 and stats["others"]["kept"] == 0
    assert stats["building"]["kept"] == 2 and stats["vegetation"]["kept"] == 2


def main():
    tests = [test_grid_subsample, test_hag_flat_and_sloped, test_features_recipe, test_label_mapping]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
