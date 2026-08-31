"""Phase 6 verification: full-tile inference -> classified .laz, with checks + viz.

Uses the Phase-5 smoke model (barely trained -- we verify the PIPELINE, not quality).
Run:  python3 scripts/check_infer.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import laspy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config
from randla_net.model.randla import RandLANet
from randla_net.engine.checkpoint import load_checkpoint
from randla_net.engine.inference import infer_tile
from randla_net.viz.export_las import export_las

ASPRS_NAME = {1: "others", 2: "ground", 5: "vegetation", 6: "building"}


def visualize(cfg, tile, asprs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from randla_net.config import FT2M

    las = laspy.read(str(cfg.orig_laz(tile)))
    x = (np.asarray(las.x) - las.header.x_min) * FT2M
    y = (np.asarray(las.y) - las.header.y_min) * FT2M
    s = np.random.RandomState(0).choice(x.size, min(500000, x.size), replace=False)
    code2col = {1: "#ff7f0e", 2: "#8c6d31", 5: "#2ca02c", 6: "#d62728"}
    col = np.array([code2col.get(int(c), "#000000") for c in asprs[s]])
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(x[s], y[s], c=col, s=0.4)
    ax.set_aspect("equal")
    ax.set_title(f"{tile} predicted classes (smoke model)\nbrown=ground green=veg red=building orange=others")
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    plt.tight_layout()
    out = cfg.out_dir / f"phase6_{tile}_pred.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    cfg = Config(num_points=16384)
    tile = "BS842712"
    device = "cpu"
    model = RandLANet(cfg, in_dim=3 + cfg.feat.in_dim, num_out=4)
    load_checkpoint(Path(__file__).resolve().parents[1] / "outputs" / "smoke" / "best.pth", model)

    t0 = time.time()
    full_pred, seen, nb = infer_tile(cfg, model, cfg.cache_dir / f"{tile}.npz", device,
                                     min_votes=1, max_blocks=600, log_every=100)
    print(f"inference: {nb} blocks in {time.time()-t0:.0f}s | min_seen={seen.min()} covered={100*np.mean(seen>0):.1f}%")

    out_laz = cfg.out_dir / f"{tile}_pred.laz"
    asprs = export_las(cfg, cfg.orig_laz(tile), full_pred, out_laz)

    # --- checks ---
    orig = laspy.read(str(cfg.orig_laz(tile)))
    back = laspy.read(str(out_laz))
    checks = []
    checks.append(("pred length == points", full_pred.shape[0] == orig.header.point_count))
    checks.append(("exported point count", back.header.point_count == orig.header.point_count))
    uniq = set(np.unique(np.asarray(back.classification)).tolist())
    checks.append(("ASPRS codes subset {1,2,5,6}", uniq <= {1, 2, 5, 6}))
    # reproject correctness: full_pred = sub_pred[inv]
    d = np.load(cfg.cache_dir / f"{tile}.npz")
    checks.append(("reproject len == inv len", full_pred.shape[0] == d["inv"].shape[0]))

    print("\ndistribution:")
    u, c = np.unique(asprs, return_counts=True)
    for k, v in zip(u, c):
        print(f"   {ASPRS_NAME.get(int(k), k):11s}({k}): {100*v/asprs.size:5.1f}%")
    print()
    ok = all(p for _, p in checks)
    for name, p in checks:
        print(f"   [{'OK' if p else 'FAIL'}] {name}")
    viz = visualize(cfg, tile, asprs)
    print(f"\n[laz] {out_laz}\n[viz] {viz}\nPHASE6 {'OK' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
