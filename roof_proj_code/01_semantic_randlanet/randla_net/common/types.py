"""Typed containers for a training block. Replaces the TF1 reference's opaque
`flat_inputs` positional list -- everything is named and `.to(device)`-aware.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultiScaleIndices:
    """Per-layer neighbourhood indices for the RandLA-Net encoder/decoder.

    Lists of length num_layers. Tensors are (B, N_l, ...) after collate, or
    (N_l, ...) for a single Sample.
    """

    neighbors: list   # [L] (.., N_l, k)        KNN at each layer
    sub_idx: list     # [L] (.., N_{l+1}, k)    pooling neighbours of kept points
    interp_idx: list  # [L] (.., N_l, 1)        nearest kept point (decoder upsample)

    def to(self, device) -> "MultiScaleIndices":
        return MultiScaleIndices(
            [t.to(device) for t in self.neighbors],
            [t.to(device) for t in self.sub_idx],
            [t.to(device) for t in self.interp_idx],
        )


@dataclass
class Sample:
    """One training block (numpy, CPU, produced by Dataset.__getitem__)."""

    coords: object       # (N,3) float32  block-centered xyz (layer 0)
    feats: object        # (N,C) float32  recipe channels
    labels: object       # (N,)  int64    internal codes, 0=unlabeled/ignore
    ms: MultiScaleIndices
    point_idx: object    # (N,)  int64    index into the tile sub-cloud (for reprojection)
    tile: str


@dataclass
class Batch:
    coords: torch.Tensor     # (B,N,3)
    feats: torch.Tensor      # (B,N,C)
    labels: torch.Tensor     # (B,N)
    ms: MultiScaleIndices    # each list tensor (B, N_l, ...)
    point_idx: torch.Tensor  # (B,N)
    tiles: list

    def to(self, device) -> "Batch":
        return Batch(
            self.coords.to(device),
            self.feats.to(device),
            self.labels.to(device),
            self.ms.to(device),
            self.point_idx.to(device),
            self.tiles,
        )
