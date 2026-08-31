"""Phase 3 verification: shape contracts + a sampled-block visualization.

Run:  python3 scripts/check_dataset.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config
from randla_net.data.dataset import ColumbusDataset
from randla_net.data.collate import collate

CODE_COLOR = ["#cccccc", "#ff7f0e", "#8c6d31", "#2ca02c", "#d62728"]


def layer_sizes(n, ratios):
    sizes = [n]
    for r in ratios:
        sizes.append(max(1, sizes[-1] // r))
    return sizes  # len = num_layers+1


def check_contracts(cfg: Config, b) -> None:
    B, N, C = b.coords.shape
    assert b.coords.shape == (B, N, 3)
    assert b.feats.shape == (B, N, cfg.feat.in_dim), b.feats.shape
    assert b.labels.shape == (B, N)
    assert b.labels.min() >= 0 and b.labels.max() <= 4, (b.labels.min(), b.labels.max())
    assert b.point_idx.shape == (B, N)
    sizes = layer_sizes(N, cfg.sub_ratios)
    for i in range(cfg.num_layers):
        ni, ni1 = sizes[i], sizes[i + 1]
        nb, sub, up = b.ms.neighbors[i], b.ms.sub_idx[i], b.ms.interp_idx[i]
        assert nb.shape == (B, ni, cfg.k_n), (i, nb.shape, (B, ni, cfg.k_n))
        assert sub.shape == (B, ni1, cfg.k_n), (i, sub.shape, (B, ni1, cfg.k_n))
        assert up.shape == (B, ni, 1), (i, up.shape, (B, ni, 1))
        assert nb.dtype == torch.int64 and up.dtype == torch.int64
        assert nb.min() >= 0 and nb.max() < ni, (i, "neigh oob", nb.max(), ni)
        assert up.min() >= 0 and up.max() < ni1, (i, "interp oob", up.max(), ni1)
        assert sub.min() >= 0 and sub.max() < ni, (i, "sub oob", sub.max(), ni)
    b.to("cpu")  # .to() moves every nested tensor without error
    print(f"  contracts OK: B={B} N={N} C={C} layer sizes={sizes}")


def visualize(cfg: Config, b) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    xy = b.coords[0].numpy()
    lab = b.labels[0].numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(xy[:, 0], xy[:, 1], c=lab, s=1.0,
               cmap=ListedColormap(CODE_COLOR), vmin=-0.5, vmax=4.5)
    ax.set_aspect("equal")
    ax.set_title(f"sampled block ({b.tiles[0]})  colored by label")
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    plt.tight_layout()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / "phase3_block.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    cfg = Config(num_points=16384)  # smaller block for fast CPU verification; 65536 on the 5090
    ds = ColumbusDataset(cfg, cfg.train_tiles, split="train", steps=8, seed=0)
    dl = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate, num_workers=0)

    t0 = time.time()
    batches = []
    for k, b in enumerate(dl):
        check_contracts(cfg, b)
        names = {c: int((b.labels == c).sum()) for c in range(5)}
        print(f"  batch {k} tiles={b.tiles} label counts={names}")
        batches.append(b)
        if k >= 1:
            break
    print(f"  pulled {len(batches)} batches of {cfg.batch_size} in {time.time()-t0:.1f}s")
    out = visualize(cfg, batches[0])
    print(f"[viz] {out}\nPHASE3 OK")


if __name__ == "__main__":
    main()
