"""Unit tests for Phase 4 model ops + loss, on synthetic data with known answers.

Catches semantic bugs that overfit-one-batch can mask (capacity hides wiring sins).
Run:  python3 tests/test_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randla_net.config import Config
from randla_net.model.blocks import (
    gather_neighbors, nearest_interp, random_sample, relative_pos_encoding,
)
from randla_net.model.randla import RandLANet
from randla_net.losses.weighted_ce import masked_weighted_ce
from randla_net.common.types import Sample
from randla_net.data.collate import collate
from randla_net.data.multiscale import build_multiscale


def test_gather_neighbors():
    feats = torch.tensor([[[0., 0], [1, 10], [2, 20], [3, 30]]])  # (1,4,2)
    idx = torch.tensor([[[0, 2], [1, 3]]])                        # (1,2,2)
    out = gather_neighbors(feats, idx)                            # (1,2,2,2)
    assert out.shape == (1, 2, 2, 2)
    assert torch.allclose(out[0, 0, 0], feats[0, 0])  # point0 neighbor idx0
    assert torch.allclose(out[0, 0, 1], feats[0, 2])  # point0 neighbor idx2
    assert torch.allclose(out[0, 1, 1], feats[0, 3])  # point1 neighbor idx3


def test_nearest_interp():
    feats = torch.tensor([[[5.], [7.]]])                # (1,2,1) sub features
    interp = torch.tensor([[[0], [1], [0]]])            # (1,3,1)
    out = nearest_interp(feats, interp)
    assert out.flatten().tolist() == [5., 7., 5.]


def test_random_sample():
    feats = torch.tensor([[[1.], [2.], [3.], [4.]]])    # (1,4,1)
    sub = torch.tensor([[[0, 1], [2, 3]]])              # kept0 max(1,2)=2 ; kept1 max(3,4)=4
    out = random_sample(feats, sub)
    assert out.flatten().tolist() == [2., 4.]


def test_relative_pos_encoding():
    xyz = torch.tensor([[[0., 0, 0], [1, 0, 0]]])       # (1,2,3)
    idx = torch.tensor([[[0, 1], [1, 0]]])
    rp = relative_pos_encoding(xyz, idx)                # (1,2,2,10)
    # point0, neighbor=point1: center(0,0,0) nb(1,0,0) rel(-1,0,0) dist 1
    assert torch.allclose(rp[0, 0, 1, 0:3], torch.tensor([0., 0, 0]))
    assert torch.allclose(rp[0, 0, 1, 3:6], torch.tensor([1., 0, 0]))
    assert torch.allclose(rp[0, 0, 1, 6:9], torch.tensor([-1., 0, 0]))
    assert abs(rp[0, 0, 1, 9].item() - 1.0) < 1e-5


def test_masked_loss():
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[0, 2, 4]])                  # ignore, ground(->1), building(->3)
    logits[0, 1, 1] = 10.0
    logits[0, 2, 3] = 10.0  # correct on the two labeled points
    assert masked_weighted_ce(logits, labels).item() < 0.01
    logits[0, 2, 3] = 0.0
    logits[0, 2, 0] = 10.0  # make building point wrong
    assert masked_weighted_ce(logits, labels).item() > 1.0
    # all-ignore batch -> zero, no crash
    assert masked_weighted_ce(logits, torch.zeros(1, 3, dtype=torch.long)).item() == 0.0


def test_model_forward_and_grad():
    cfg = Config(num_points=2048, batch_size=2)
    rng = np.random.RandomState(3)
    samples = []
    for i in range(cfg.batch_size):
        coords = rng.normal(size=(cfg.num_points, 3)).astype(np.float32)
        feats = rng.normal(size=(cfg.num_points, cfg.feat.in_dim)).astype(np.float32)
        labels = rng.randint(0, 5, size=cfg.num_points, dtype=np.int64)
        ms = build_multiscale(coords, cfg.num_layers, cfg.k_n, cfg.sub_ratios)
        samples.append(Sample(coords, feats, labels, ms,
                              np.arange(cfg.num_points, dtype=np.int64), f"synthetic-{i}"))
    batch = collate(samples)
    model = RandLANet(cfg, in_dim=3 + cfg.feat.in_dim, num_out=4)

    model.eval()
    with torch.no_grad():
        o1 = model(batch)
        o2 = model(batch)
    assert o1.shape == (2, 2048, 4), o1.shape
    assert torch.allclose(o1, o2), "eval-mode forward must be deterministic"

    model.train()
    loss = masked_weighted_ce(model(batch), batch.labels)
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"params with no gradient: {dead}"


def main():
    tests = [test_gather_neighbors, test_nearest_interp, test_random_sample,
             test_relative_pos_encoding, test_masked_loss, test_model_forward_and_grad]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
