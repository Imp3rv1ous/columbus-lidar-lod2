"""Production training CLI (run on the RTX 5090).

  python3 scripts/train.py --epochs 100 --device cuda --workers 32 --batch-size 8

Trains route A (RandLA-Net encoder-decoder + masked weighted CE) on the 4
training tiles. best.pth is selected on an EVAL-MODE mIoU (running stats) so the
saved model reflects inference behaviour rather than train-mode BatchNorm statistics.
Checkpoints + metrics.csv
land in --out; if RANDLA_REPORT is set, per-epoch per-class IoU is appended there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config
from randla_net.data.dataset import ColumbusDataset
from randla_net.data.collate import collate
from randla_net.model.randla import RandLANet
from randla_net.losses.weighted_ce import inverse_sqrt_weights
from randla_net.engine.trainer import Trainer
from randla_net.common.runlog import RunLogger


def _worker_init(worker_id: int) -> None:
    """Give each forked DataLoader worker a unique sampler seed (no duplicate blocks)."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.reseed(ds._seed * 1000 + worker_id + 1)


def class_counts(cfg: Config, tiles) -> dict:
    counts = {}
    for t in tiles:
        lab = np.load(cfg.cache_dir / f"{t}_labels.npy")
        for c in (1, 2, 3, 4):
            counts[c] = counts.get(c, 0) + int((lab == c).sum())
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--num-points", type=int, default=65536)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=100,
                    help="eval-mode blocks per epoch for honest best.pth selection")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="checkpoint dir; default = outputs/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--building-weight", type=float, default=1.0,
                    help="extra multiplier on building weight; 1.0 = natural inverse-sqrt "
                         "(building is already up-weighted by inverse frequency).")
    ap.add_argument("--holdout-tile", default=None,
                    help="hold this training tile out as a TRUE (unseen) validation tile for a "
                         "real generalization metric; default None = eval on all train tiles "
                         "(a train-tile eval-mode selection metric, not generalization).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = Config(num_points=args.num_points, batch_size=args.batch_size,
                 steps_per_epoch=args.steps, max_epochs=args.epochs)
    args.out = args.out or str(cfg.out_dir)

    all_train = list(cfg.train_tiles)
    if args.holdout_tile:
        train_tiles = [t for t in all_train if t != args.holdout_tile]
        val_tiles = [args.holdout_tile]                 # a TRUE unseen-tile metric
        val_kind = f"holdout {args.holdout_tile} (generalization)"
    else:
        train_tiles, val_tiles = all_train, all_train   # train-tile eval-mode selection
        val_kind = "train tiles (eval-mode selection, not generalization)"

    counts = class_counts(cfg, train_tiles)
    w = inverse_sqrt_weights(counts, 4)
    w[3] *= args.building_weight   # building = output index 3
    weights = [round(x, 2) for x in w.tolist()]
    print(f"device={device} | train={train_tiles} | val={val_tiles} [{val_kind}]")
    print(f"class counts={counts}")
    print(f"class weights (others/ground/veg/building)={weights} (building x{args.building_weight})")

    train_ds = ColumbusDataset(cfg, train_tiles, "train", steps=args.steps, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, collate_fn=collate,
                          num_workers=args.workers, persistent_workers=args.workers > 0,
                          worker_init_fn=_worker_init)
    # eval-mode loader (no augmentation): an honest selection metric that exposes any
    # train/eval (BatchNorm) gap. Non-persistent so workers respawn each epoch ->
    # sampler resets to the same per-worker seed -> a STATIONARY metric comparable
    # across epochs (apples-to-apples selection).
    val_workers = min(8, args.workers)
    val_ds = ColumbusDataset(cfg, val_tiles, "val", steps=args.eval_steps, seed=args.seed + 777)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate,
                        num_workers=val_workers, persistent_workers=False,
                        worker_init_fn=_worker_init)

    model = RandLANet(cfg, in_dim=3 + cfg.feat.in_dim, num_out=4)
    print(f"model params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    logger = RunLogger.from_env()
    if logger.enabled:
        logger.header("Train config")
        logger.kv({
            "train_tiles": ", ".join(train_tiles),
            "val": val_kind,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "num_points": args.num_points, "steps/epoch": args.steps,
            "eval_steps": args.eval_steps, "workers": args.workers,
            "building_weight": args.building_weight,
            "class counts (1..4)": counts,
            "class weights (oth/grd/veg/bld)": weights,
            "out": args.out,
        })

    tr = Trainer(cfg, model, train_dl, device, weight=w, val_loader=val_dl,
                 out_dir=args.out, num_classes=4, logger=logger)
    best = tr.fit(args.epochs)
    print(f"done. best eval mIoU={best:.3f} | checkpoints in {args.out}")


if __name__ == "__main__":
    main()
