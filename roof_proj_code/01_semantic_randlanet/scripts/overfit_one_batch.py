"""Phase 4 sanity gate: overfit a single batch.

If the whole pipeline (sample -> forward -> masked loss -> backward) is wired
correctly and the model has capacity, loss -> ~0 and train accuracy -> ~1.0.
Run:  python3 scripts/overfit_one_batch.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config
from randla_net.data.dataset import ColumbusDataset
from randla_net.data.collate import collate
from randla_net.model.randla import RandLANet
from randla_net.losses.weighted_ce import masked_weighted_ce

NAME = {0: "unlabeled", 1: "others", 2: "ground", 3: "veg", 4: "building"}


def main() -> None:
    torch.manual_seed(0)
    cfg = Config(num_points=8192, batch_size=2)
    # use the GPU if present (CPU here cost ~1.5 h on the 5090 run); override with RANDLA_DEVICE
    device = os.environ.get("RANDLA_DEVICE",
                            "cuda" if torch.cuda.is_available() else "cpu")

    ds = ColumbusDataset(cfg, cfg.train_tiles, split="val", steps=10, seed=0)
    batch = collate([ds[0] for _ in range(cfg.batch_size)]).to(device)

    lab = batch.labels.reshape(-1)
    dist = {NAME[c]: int((lab == c).sum()) for c in range(5)}
    print(f"batch: B={cfg.batch_size} N={cfg.num_points} label dist={dist}")

    model = RandLANet(cfg, in_dim=3 + cfg.feat.in_dim, num_out=4).to(device)
    model.drop.p = 0.0  # clean overfit signal
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    t0 = time.time()
    last_loss, last_acc = None, None
    for step in range(201):
        opt.zero_grad()
        logits = model(batch)
        loss = masked_weighted_ce(logits, batch.labels)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == 200:
            with torch.no_grad():
                valid = lab != 0
                pred = logits.reshape(-1, 4).argmax(-1)[valid]
                tgt = lab[valid] - 1
                acc = (pred == tgt).float().mean().item()
            last_loss, last_acc = loss.item(), acc
            print(f"  step {step:3d}  loss {loss.item():.4f}  train_acc {acc:.3f}")
    print(f"\n{201} steps in {time.time()-t0:.1f}s")
    ok = last_loss < 0.1 and last_acc > 0.95
    print(f"OVERFIT {'OK (pipeline wired correctly)' if ok else 'FAILED -- investigate'}: "
          f"loss={last_loss:.4f} acc={last_acc:.3f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
