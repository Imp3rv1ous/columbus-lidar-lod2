"""Write predictions to LAZ, remapping internal labels to ASPRS class codes."""
from __future__ import annotations

import laspy
import numpy as np


def export_las(cfg, orig_path, full_pred_internal: np.ndarray, out_path) -> np.ndarray:
    asprs_map = dict(cfg.classes.to_asprs)                  # {1:1, 2:2, 3:5, 4:6}
    lut = np.array([asprs_map.get(c, 1) for c in range(5)])  # index by internal code 0..4
    asprs = lut[full_pred_internal].astype(np.uint8)

    las = laspy.read(str(orig_path))
    if asprs.shape[0] != las.header.point_count:
        raise ValueError(f"pred {asprs.shape[0]} != points {las.header.point_count}")
    las.classification = asprs
    las.write(str(out_path))
    return asprs
