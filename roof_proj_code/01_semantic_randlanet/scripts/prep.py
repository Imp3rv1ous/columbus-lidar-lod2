"""Phase 1 preprocessing: read tiles -> HAG -> features -> grid subsample -> cache.

Run:  python3 scripts/prep.py            # processes the 4 training tiles
Outputs per tile:  cache/<tile>.npz   and a verification figure in outputs/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `randla_net` importable

from randla_net.config import Config
from randla_net.prep.read_las import read_tile
from randla_net.prep.hag import compute_hag
from randla_net.prep.features import build_features
from randla_net.prep.grid_subsample import grid_subsample
from randla_net.common.runlog import RunLogger


def process_tile(cfg: Config, tile: str) -> dict:
    t0 = time.time()
    src = cfg.orig_laz(tile)
    data = read_tile(src)
    hag = compute_hag(data["xyz"], data["classification"], cfg.hag_cell, cfg.hag_knn)
    feats = build_features(data, hag, cfg.feat)
    sub_xyz, sub_feats, inv = grid_subsample(data["xyz"], feats, cfg.sub_grid)

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.cache_dir / f"{tile}.npz",
        sub_xyz=sub_xyz.astype(np.float32),
        sub_feats=sub_feats.astype(np.float32),
        inv=inv.astype(np.int32),                         # full -> sub (reprojection index)
        full_class=data["classification"].astype(np.int16),
        origin=data["origin"].astype(np.float64),
        feat_channels=np.array(cfg.feat.channels),
        sub_grid=np.float64(cfg.sub_grid),
    )

    return {
        "tile": tile,
        "n_full": data["xyz"].shape[0],
        "n_sub": sub_xyz.shape[0],
        "hag": hag,
        "feats": feats,
        "sub_xyz": sub_xyz,
        "sub_feats": sub_feats,
        "ground_frac": float((data["classification"] == 2).mean()),
        "secs": time.time() - t0,
    }


def print_stats(cfg: Config, r: dict) -> None:
    ch = list(cfg.feat.channels)
    hag = r["hag"]
    print(f"\n=== {r['tile']}  ({r['secs']:.1f}s) ===")
    print(f"  points: full {r['n_full']:,} -> sub {r['n_sub']:,} "
          f"({100*r['n_sub']/r['n_full']:.1f}% kept) | ground={100*r['ground_frac']:.0f}%")
    print(f"  HAG (m): p5={np.percentile(hag,5):.2f} p50={np.percentile(hag,50):.2f} "
          f"p95={np.percentile(hag,95):.2f} min={hag.min():.2f} max={hag.max():.2f}")
    print("  feature channels (sub-cloud min..max):")
    for i, c in enumerate(ch):
        col = r["sub_feats"][:, i]
        print(f"     {c:20s} {col.min():8.3f} .. {col.max():8.3f}  mean={col.mean():.3f}")


def visualize(cfg: Config, r: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sx = r["sub_xyz"]
    ch = list(cfg.feat.channels)
    hag_i = ch.index("hag")
    int_i = ch.index("intensity_norm")

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    s = 0.4
    p0 = ax[0].scatter(sx[:, 0], sx[:, 1], c=np.clip(r["sub_feats"][:, hag_i], 0, 20), s=s, cmap="turbo")
    ax[0].set_title(f"{r['tile']}  HAG (m) [subsampled]\nbuildings/trees rise, ground~0")
    plt.colorbar(p0, ax=ax[0], shrink=0.7, label="HAG (m)")

    p1 = ax[1].scatter(sx[:, 0], sx[:, 1], c=r["sub_feats"][:, int_i], s=s, cmap="gray")
    ax[1].set_title("intensity_norm [subsampled]")
    plt.colorbar(p1, ax=ax[1], shrink=0.7, label="intensity (0-1)")

    ax[2].scatter(sx[:, 0], sx[:, 1], c="k", s=s)
    ax[2].set_title(f"subsampled cloud ({r['n_sub']:,} pts @ {cfg.sub_grid} m)")

    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("m East")
        a.set_ylabel("m North")
    plt.tight_layout()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / f"phase1_{r['tile']}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    return out


def all_tiles(cfg: Config) -> list:
    import glob
    return sorted(Path(p).stem.split("_B19_")[1] for p in glob.glob(str(cfg.data_dir / "*.laz")))


def main() -> None:
    cfg = Config()
    tiles = sys.argv[1:] or list(cfg.train_tiles)   # pass tile names, or "all", default = training tiles
    if tiles == ["all"]:
        tiles = all_tiles(cfg)
    print(f"feature recipe: {cfg.feat.channels}  (in_dim={cfg.feat.in_dim})")
    print(f"sub_grid={cfg.sub_grid} m | classes={cfg.classes.num_classes} (+ignore) | tiles={tiles}")
    results = []
    for tile in tiles:
        r = process_tile(cfg, tile)
        print_stats(cfg, r)
        results.append(r)

    logger = RunLogger.from_env()
    if logger.enabled:
        logger.header("Prep (per-tile)")
        logger.table_header(["tile", "n_full", "n_sub", "kept%", "ground%",
                             "HAG p50", "HAG p95", "HAG max"])
        for r in results:
            hag = r["hag"]
            logger.row([r["tile"], f"{r['n_full']:,}", f"{r['n_sub']:,}",
                        f"{100*r['n_sub']/r['n_full']:.1f}", f"{100*r['ground_frac']:.0f}",
                        f"{np.percentile(hag,50):.2f}", f"{np.percentile(hag,95):.2f}",
                        f"{hag.max():.2f}"])
        logger.line()
        low = [r["tile"] for r in results if r["ground_frac"] < 0.05]
        if low:
            logger.warn(f"tiles with <5% ground points — HAG may be unreliable there: {low}")

    # visualize the dense-urban tile for eyeballing
    viz_tile = "BS842712" if any(r["tile"] == "BS842712" for r in results) else results[0]["tile"]
    out = visualize(cfg, next(r for r in results if r["tile"] == viz_tile))
    print(f"\n[viz] {out}")
    print(f"[cache] {cfg.cache_dir}")


if __name__ == "__main__":
    main()
