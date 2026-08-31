"""Composable block transforms (callables operating on (coords, feats) numpy).

ALS is gravity-aligned: yaw-only rotation, never tumble. HAG is a precomputed
feature and is left untouched (only z-jitter would perturb it, which we avoid).
"""
from __future__ import annotations

import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coords, feats):
        for t in self.transforms:
            coords, feats = t(coords, feats)
        return coords, feats


class RecenterBlock:
    """Center the block on its xyz centroid (xy AND z).

    z is centered too — absolute in-block z = height above the tile's lowest
    point, which drifts with each tile's terrain elevation and made the net latch
    onto tile-specific height bands (a cross-tile generalization leak). HAG and the
    rpe rel/dist channels stay drift-free; the rpe center/nb absolute-coord channels
    and coords-as-features do get block-centered too — intended, that is exactly
    what removes the tile-specific z drift. Applied identically in train and
    inference, so the two paths stay consistent.
    """

    def __call__(self, coords, feats):
        c = coords - coords.mean(axis=0)
        return c, feats


class RandomRotateZ:
    def __init__(self, rng):
        self.rng = rng

    def __call__(self, coords, feats):
        a = self.rng.uniform(0, 2 * np.pi)
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        return coords @ R.T, feats


class JitterXY:
    def __init__(self, rng, sigma=0.02, clip=0.05):
        self.rng, self.sigma, self.clip = rng, sigma, clip

    def __call__(self, coords, feats):
        n = np.clip(self.rng.randn(*coords.shape) * self.sigma, -self.clip, self.clip)
        n[:, 2] = 0.0  # do not jitter z (would corrupt the HAG/height prior)
        return coords + n, feats


class RandomScale:
    """Isotropic xy scaling — regularizes against overfitting to training-tile scale."""

    def __init__(self, rng, lo=0.9, hi=1.1):
        self.rng, self.lo, self.hi = rng, lo, hi

    def __call__(self, coords, feats):
        s = self.rng.uniform(self.lo, self.hi)
        c = coords.copy()
        c[:, :2] *= s          # scale xy only; z stays (HAG feature is separate)
        return c, feats


def build_transforms(augment: bool, rng):
    ts = [RecenterBlock()]
    if augment:
        ts += [RandomRotateZ(rng), RandomScale(rng), JitterXY(rng)]
    return Compose(ts)
