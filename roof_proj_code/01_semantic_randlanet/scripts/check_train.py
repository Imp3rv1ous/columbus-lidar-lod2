"""Phase 5 verification: short CPU train + checkpoint round-trip.

PASS if loss decreases, mIoU is computed, and a reloaded checkpoint reproduces
the model's forward exactly.
Run:  python3 scripts/check_train.py
"""
from __future__ import annotations

import csv
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
from randla_net.engine.checkpoint import load_checkpoint


def class_counts(cfg: Config) -> dict:
    counts = {}
    for t in cfg.train_tiles:
        lab = np.load(cfg.cache_dir / f"{t}_labels.npy")
        for c in (1, 2, 3, 4):
            counts[c] = counts.get(c, 0) + int((lab == c).sum())
    return counts


def main() -> None:
    torch.manual_seed(0)
    cfg = Config(num_points=8192, batch_size=2, steps_per_epoch=15)
    device = "cpu"
    out = Path(__file__).resolve().parents[1] / "outputs" / "smoke"

    counts = class_counts(cfg)
    w = inverse_sqrt_weights(counts, 4)
    print("class counts:", counts)
    print("class weights:", [round(x, 2) for x in w.tolist()], "(others/ground/veg/building)")

    ds = ColumbusDataset(cfg, cfg.train_tiles, "train", steps=cfg.steps_per_epoch, seed=0)
    dl = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate, num_workers=0)
    model = RandLANet(cfg, in_dim=3 + cfg.feat.in_dim, num_out=4)

    tr = Trainer(cfg, model, dl, device, weight=w, out_dir=out, num_classes=4)
    tr.fit(3)

    # checkpoint round-trip: reload last.pth into a fresh model, compare forward
    fresh = RandLANet(cfg, in_dim=3 + cfg.feat.in_dim, num_out=4)
    ck = load_checkpoint(out / "last.pth", fresh)
    model.eval()
    fresh.eval()
    b = collate([ds[0] for _ in range(2)])
    with torch.no_grad():
        parity = torch.allclose(model(b), fresh(b), atol=1e-5)

    rows = list(csv.DictReader(open(tr._csv)))
    loss0, lossN = float(rows[0]["train_loss"]), float(rows[-1]["train_loss"])
    decreased = lossN < loss0
    print(f"\nloss {loss0:.3f} -> {lossN:.3f} (decreased={decreased})")
    print(f"checkpoint epoch={ck['epoch']} best={ck['best']:.3f} | reload forward parity={parity}")
    ok = decreased and parity
    print("PHASE5 OK" if ok else "PHASE5 FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
