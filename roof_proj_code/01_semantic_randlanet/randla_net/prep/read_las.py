"""Read a LAS/LAZ tile -> metric, tile-centered point arrays.

The only module that touches laspy / coordinate units. Everything downstream
works in METERS, centered on the tile's min corner (so coords are small and
float32-safe). The metric origin is returned so we can reproject back to the
original CRS when writing predictions.
"""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from ..config import FT2M


def read_tile(path: str | Path) -> dict:
    las = laspy.read(str(path))

    # absolute coords are in US survey feet -> meters
    xyz_m = np.column_stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
    ).astype(np.float64) * FT2M

    origin = xyz_m.min(axis=0)
    xyz = xyz_m - origin  # tile-centered metric coords, in [0, ~381]

    return {
        "xyz": xyz,                                                    # (N,3) float64, meters
        "origin": origin,                                             # (3,)  meters (for reprojection)
        "intensity": np.asarray(las.intensity).astype(np.float64),   # (N,)
        "return_number": np.asarray(las.return_number).astype(np.float64),
        "num_returns": np.asarray(las.number_of_returns).astype(np.float64),
        "classification": np.asarray(las.classification).astype(np.int64),  # ASPRS codes
    }
