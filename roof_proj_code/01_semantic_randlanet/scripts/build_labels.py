"""Phase 2: build & cache sparse training labels, with verification viz.

Run:  python3 scripts/build_labels.py
Outputs:  cache/<tile>_labels.npy  +  outputs/phase2_<tile>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config
from randla_net.prep.labels import build_labels
from randla_net.common.runlog import RunLogger

CODE_NAME = {0: "unlabeled", 1: "others", 2: "ground", 3: "vegetation", 4: "building"}
CODE_COLOR = {0: "#cccccc", 1: "#ff7f0e", 2: "#8c6d31", 3: "#2ca02c", 4: "#d62728"}


def visualize(cfg: Config, tile: str, labels: np.ndarray) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    sx = np.load(cfg.cache_dir / f"{tile}.npz")["sub_xyz"]
    cmap = ListedColormap([CODE_COLOR[c] for c in range(5)])

    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(sx[:, 0], sx[:, 1], c=labels, s=0.4, cmap=cmap, vmin=-0.5, vmax=4.5)
    ax.set_aspect("equal")
    ax.set_xlabel("m East")
    ax.set_ylabel("m North")
    ax.set_title(f"{tile}  training labels  (gray=unlabeled/ignore)")
    cb = plt.colorbar(sc, ticks=range(5), shrink=0.8)
    cb.ax.set_yticklabels([CODE_NAME[c] for c in range(5)])
    plt.tight_layout()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / f"phase2_{tile}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    cfg = Config()
    logger = RunLogger.from_env()
    rows = []
    for tile in cfg.train_tiles:
        labels, stats, guard = build_labels(cfg, tile)
        np.save(cfg.cache_dir / f"{tile}_labels.npy", labels.astype(np.int8))
        n = labels.size
        print(f"\n=== {tile}  (guard={guard:.2f} m) ===")
        counts = {c: int((labels == c).sum()) for c in range(5)}
        for c in range(5):
            print(f"   {c} {CODE_NAME[c]:11s}: {counts[c]:>9,} ({100*counts[c]/n:5.1f}%)")
        labeled = n - counts[0]
        print(f"   labeled (1-4): {labeled:,} ({100*labeled/n:.1f}% of sub cloud)")
        for name, s in stats.items():
            print(f"   user {name:11s}: kept {s['kept']:>7,}  rejected {s['rejected']:,}")
        rows.append([tile, f"{guard:.2f}", f"{counts[1]:,}", f"{counts[2]:,}",
                     f"{counts[3]:,}", f"{counts[4]:,}", f"{100*labeled/n:.1f}"])

    if logger.enabled:
        logger.header("Labels (per training tile, sub-cloud counts)")
        logger.table_header(["tile", "guard m", "others", "ground", "veg", "building", "labeled%"])
        for r in rows:
            logger.row(r)
        logger.line()

    viz_tile = cfg.train_tiles[0]
    out = visualize(cfg, viz_tile, np.load(cfg.cache_dir / f"{viz_tile}_labels.npy"))
    print(f"\n[viz] {out}")


if __name__ == "__main__":
    main()
