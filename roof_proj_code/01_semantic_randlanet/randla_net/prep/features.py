"""Recipe-driven per-point feature assembly.

The recipe (config.FeatureRecipe.channels) is the single switch that selects
which channels are fed to the network. Adding/removing a channel is a one-line
configuration change rather than a separate implementation for each feature set.
"""
from __future__ import annotations

import numpy as np

from ..config import FeatureRecipe


def norm_intensity(intensity: np.ndarray) -> np.ndarray:
    """Per-tile robust min-max to [0,1] (clip to 2nd/98th percentile)."""
    lo, hi = np.percentile(intensity, [2, 98])
    return np.clip((intensity - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def build_features(tile: dict, hag: np.ndarray, recipe: FeatureRecipe) -> np.ndarray:
    """Return (N, in_dim) feature matrix in the exact channel order of the recipe."""
    available = {
        "intensity_norm": lambda: norm_intensity(tile["intensity"]),
        "return_number_norm": lambda: np.clip(tile["return_number"] / 7.0, 0.0, 1.0),
        "num_returns_norm": lambda: np.clip(tile["num_returns"] / 7.0, 0.0, 1.0),
        "hag": lambda: hag.astype(np.float64),
    }
    cols = []
    for ch in recipe.channels:
        if ch not in available:
            raise KeyError(f"unknown feature channel: {ch!r}")
        cols.append(available[ch]())
    return np.column_stack(cols)
