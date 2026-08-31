"""Production inference CLI: classify tiles -> .laz (run on the 5090, or anywhere).

  python3 scripts/infer.py --tiles BS841708
  python3 scripts/infer.py --all

Requires each tile to be prepped first (scripts/prep.py <tile> or `all`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config
from randla_net.model.randla import RandLANet
from randla_net.engine.checkpoint import load_checkpoint
from randla_net.engine.inference import infer_tile
from randla_net.viz.export_las import export_las
from randla_net.common.runlog import RunLogger

ASPRS_NAME = {1: "others", 2: "ground", 5: "veg", 6: "building"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path; default = results/best.pth")
    ap.add_argument("--tiles", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--num-points", type=int, default=65536)
    ap.add_argument("--min-votes", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None,
                    help="prediction directory; default = outputs/preds/")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = Config(num_points=args.num_points)
    args.ckpt = args.ckpt or str(cfg.results_dir / "best.pth")
    args.out = args.out or str(cfg.out_dir / "preds")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        from prep import all_tiles
        tiles = all_tiles(cfg)
    else:
        tiles = args.tiles or [cfg.val_tile]

    model = RandLANet(cfg, in_dim=3 + cfg.feat.in_dim, num_out=4)
    load_checkpoint(args.ckpt, model)
    model.to(device)
    print(f"device={device} ckpt={args.ckpt} tiles={tiles}")

    logger = RunLogger.from_env()
    rows = []
    for tile in tiles:
        cache = cfg.cache_dir / f"{tile}.npz"
        if not cache.exists():
            print(f"  [skip] {tile}: no cache -- run `python3 scripts/prep.py {tile}` first")
            continue
        t0 = time.time()
        full_pred, seen, nb = infer_tile(cfg, model, cache, device, min_votes=args.min_votes)
        out_laz = out_dir / f"{tile}_pred.laz"
        asprs = export_las(cfg, cfg.orig_laz(tile), full_pred, out_laz)
        u, c = np.unique(asprs, return_counts=True)
        dist = {ASPRS_NAME.get(int(k), k): f"{100*v/asprs.size:.1f}%" for k, v in zip(u, c)}
        cov = 100 * np.mean(seen > 0)
        print(f"  {tile}: {nb} blocks {time.time()-t0:.0f}s covered={cov:.0f}% "
              f"-> {out_laz.name}  {dist}")
        pct = {ASPRS_NAME[k]: 100 * np.sum(asprs == k) / asprs.size for k in (1, 2, 5, 6)}
        rows.append([tile, f"{asprs.size:,}", f"{pct['others']:.1f}", f"{pct['ground']:.1f}",
                     f"{pct['veg']:.1f}", f"{pct['building']:.1f}", f"{cov:.1f}"])

    if logger.enabled and rows:
        logger.header("Inference (predicted class distribution, %)")
        logger.table_header(["tile", "points", "others", "ground", "veg", "building", "coverage"])
        for r in rows:
            logger.row(r)
        logger.line("\n_BS842712 is a training tile — building% there should be clearly >0 "
                    "if the model learned & the shuffle fix works._\n")


if __name__ == "__main__":
    main()
