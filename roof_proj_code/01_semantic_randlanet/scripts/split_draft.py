"""Split a draft .laz into per-class .las files that build_labels.py consumes.

build_labels reads <tile>_building.las / _vegetation.las / _others.las and decides
the class from the FILENAME (not the stored value). Ground is filled from USGS
class-2 directly, so we do NOT export ground or the unlabeled bin.

Run this to get the starting per-class files; then correct the few wrong regions
in CloudCompare (segment the mis-labeled patch out of one file into the right one).

  python3 scripts/split_draft.py BS838711
  python3 scripts/split_draft.py BS838711 BS842712 BS841710 BS843711
"""
from __future__ import annotations

import sys
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config

# ASPRS code stored in the draft -> output label-file stem
SPLITS = {6: "building", 5: "vegetation", 1: "others"}


def run(tile: str, cfg: Config) -> None:
    draft = cfg.train_dir / tile / f"{tile}_draft.laz"
    if not draft.exists():
        print(f"  !! no draft for {tile}: {draft}")
        return
    las = laspy.read(str(draft))
    cls = np.asarray(las.classification)
    print(f"\n=== {tile} ===")
    for code, stem in SPLITS.items():
        mask = cls == code
        n = int(mask.sum())
        if n == 0:
            print(f"  {stem:<11} 0 (skipped)")
            continue
        sub = laspy.LasData(las.header)
        sub.points = las.points[mask].copy()
        out = cfg.train_dir / tile / f"{tile}_{stem}.las"
        sub.write(str(out))
        print(f"  {stem:<11} {n:>9,}  -> {out.name}")


def main() -> None:
    tiles = sys.argv[1:]
    if not tiles:
        print("usage: split_draft.py <tile> [<tile> ...]")
        sys.exit(1)
    cfg = Config()
    print(f"train_dir={cfg.train_dir}")
    for t in tiles:
        run(t, cfg)


if __name__ == "__main__":
    main()
