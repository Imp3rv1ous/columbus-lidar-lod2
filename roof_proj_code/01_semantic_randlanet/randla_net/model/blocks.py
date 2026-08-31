"""Pure-PyTorch building blocks for the RandLA-Net encoder/decoder.

All neighbour operations are plain torch.gather (CPU/MPS/CUDA identical, no
compiled ops). Tensors are (B, N, C) or (B, N, K, C); shared MLPs are 1x1 convs
expressed as Linear over the last dim.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def layer_sizes(n: int, ratios) -> list[int]:
    sizes = [n]
    for r in ratios:
        sizes.append(max(1, sizes[-1] // r))
    return sizes


class SharedMLP(nn.Module):
    """Linear over last dim + (optional) BatchNorm + LeakyReLU, shape-agnostic."""

    def __init__(self, in_c: int, out_c: int, bn: bool = True, act: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_c, out_c)
        self.bn = nn.BatchNorm1d(out_c) if bn else None
        self.act = nn.LeakyReLU(0.2) if act else None

    def forward(self, x):  # (..., in_c) -> (..., out_c)
        x = self.lin(x)
        if self.bn is not None:
            shape = x.shape
            x = self.bn(x.reshape(-1, shape[-1])).reshape(shape)
        if self.act is not None:
            x = self.act(x)
        return x


def gather_neighbors(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """feats (B,N,C), idx (B,M,K) -> (B,M,K,C)."""
    B, N, C = feats.shape
    M, K = idx.shape[1], idx.shape[2]
    flat = idx.reshape(B, M * K, 1).expand(-1, -1, C)
    out = torch.gather(feats, 1, flat)
    return out.reshape(B, M, K, C)


def relative_pos_encoding(xyz: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
    """xyz (B,N,3), neigh_idx (B,N,K) -> (B,N,K,10): [center, neighbor, rel, dist]."""
    K = neigh_idx.shape[2]
    nb = gather_neighbors(xyz, neigh_idx)                  # (B,N,K,3)
    center = xyz.unsqueeze(2).expand(-1, -1, K, -1)        # (B,N,K,3)
    rel = center - nb
    dist = torch.norm(rel, dim=-1, keepdim=True)           # (B,N,K,1)
    return torch.cat([center, nb, rel, dist], dim=-1)


class AttentivePooling(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.score = nn.Linear(in_c, in_c)
        self.mlp = SharedMLP(in_c, out_c)

    def forward(self, x):  # (B,N,K,in_c) -> (B,N,out_c)
        att = torch.softmax(self.score(x), dim=2)  # over K neighbours
        agg = (x * att).sum(dim=2)                 # (B,N,in_c)
        return self.mlp(agg)


def random_sample(feats: torch.Tensor, sub_idx: torch.Tensor) -> torch.Tensor:
    """Max-pool over neighbours of the kept points. feats (B,N,C), sub_idx (B,N',K) -> (B,N',C)."""
    nb = gather_neighbors(feats, sub_idx)
    return nb.max(dim=2).values


def nearest_interp(feats: torch.Tensor, interp_idx: torch.Tensor) -> torch.Tensor:
    """Upsample by nearest. feats (B,N',C), interp_idx (B,N,1) -> (B,N,C)."""
    B, Np, C = feats.shape
    idx = interp_idx.squeeze(-1)                  # (B,N)
    return torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, C))
