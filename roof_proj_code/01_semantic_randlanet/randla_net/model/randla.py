"""RandLA-Net: encoder (LFA + random sample) + decoder (nearest interp + skip) + head.

Faithful to the reference architecture; pure PyTorch, device-agnostic. forward()
takes a Batch and returns per-point logits over the valid classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from ..common.types import Batch
from .blocks import (
    SharedMLP,
    AttentivePooling,
    gather_neighbors,
    relative_pos_encoding,
    random_sample,
    nearest_interp,
    layer_sizes,
)


class BuildingBlock(nn.Module):
    """LocSE + Attentive Pooling, run twice (dilated)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp_xyz1 = SharedMLP(10, d_in)
        self.att1 = AttentivePooling(2 * d_in, d_out // 2)
        self.mlp_xyz2 = SharedMLP(10, d_out // 2)
        self.att2 = AttentivePooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature (B,N,d_in) -> (B,N,d_out)
        rp = relative_pos_encoding(xyz, neigh_idx)             # (B,N,K,10)
        f_xyz = self.mlp_xyz1(rp)                              # (B,N,K,d_in)
        f_nb = gather_neighbors(feature, neigh_idx)            # (B,N,K,d_in)
        f = self.att1(torch.cat([f_nb, f_xyz], dim=-1))       # (B,N,d_out//2)
        f_xyz2 = self.mlp_xyz2(rp)                             # (B,N,K,d_out//2)
        f_nb2 = gather_neighbors(f, neigh_idx)                 # (B,N,K,d_out//2)
        return self.att2(torch.cat([f_nb2, f_xyz2], dim=-1))  # (B,N,d_out)


class DilatedResBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp1 = SharedMLP(d_in, d_out // 2)
        self.bb = BuildingBlock(d_out // 2, d_out)
        self.mlp2 = SharedMLP(d_out, d_out * 2, act=False)
        self.shortcut = SharedMLP(d_in, d_out * 2, act=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, feature, xyz, neigh_idx):  # (B,N,d_in) -> (B,N,2*d_out)
        f = self.mlp1(feature)
        f = self.bb(xyz, f, neigh_idx)
        f = self.mlp2(f)
        return self.act(f + self.shortcut(feature))


class RandLANet(nn.Module):
    def __init__(self, cfg: Config, in_dim: int, num_out: int):
        super().__init__()
        self.cfg = cfg
        self.fc0 = SharedMLP(in_dim, 8)

        self.enc = nn.ModuleList()
        enc_out = []          # output channels of each encoder block (= 2*d_out[i])
        d = 8
        for i in range(cfg.num_layers):
            self.enc.append(DilatedResBlock(d, cfg.d_out[i]))
            d = cfg.d_out[i] * 2
            enc_out.append(d)
        self.bottleneck = SharedMLP(enc_out[-1], enc_out[-1])

        # decoder: at step j we upsample then concat skips[L-1-j] (= enc_out[L-1-j] channels)
        self.dec = nn.ModuleList()
        up_in = enc_out[-1]
        L = cfg.num_layers
        for j in range(L):
            skip_ch = enc_out[L - 1 - j]
            out_ch = enc_out[L - 2 - j] if (L - 2 - j) >= 0 else enc_out[0]
            self.dec.append(SharedMLP(up_in + skip_ch, out_ch))
            up_in = out_ch

        self.head = nn.Sequential(SharedMLP(up_in, 64), SharedMLP(64, 32))
        # 0.5 was too aggressive on the final 32-d head and suppressed the weak
        # rare-class (building) signal; 0.15 keeps a light regularizer.
        self.drop = nn.Dropout(0.15)
        self.fc_out = nn.Linear(32, num_out)

    def forward(self, batch: Batch) -> torch.Tensor:
        feat = torch.cat([batch.coords, batch.feats], dim=-1)  # (B,N,3+C)
        N0 = feat.shape[1]
        sizes = layer_sizes(N0, self.cfg.sub_ratios)
        xyz_l = [batch.coords[:, : sizes[i], :] for i in range(self.cfg.num_layers)]
        nbr, sub, up = batch.ms.neighbors, batch.ms.sub_idx, batch.ms.interp_idx

        f = self.fc0(feat)
        skips = []
        for i in range(self.cfg.num_layers):
            f_enc = self.enc[i](f, xyz_l[i], nbr[i])   # (B, N_i, enc_out[i])
            skips.append(f_enc)
            f = random_sample(f_enc, sub[i])           # (B, N_{i+1}, enc_out[i])
        f = self.bottleneck(f)

        for j in range(self.cfg.num_layers):
            f = nearest_interp(f, up[self.cfg.num_layers - 1 - j])    # upsample to N_{L-1-j}
            f = torch.cat([f, skips[self.cfg.num_layers - 1 - j]], dim=-1)
            f = self.dec[j](f)

        f = self.head(f)
        f = self.drop(f)
        return self.fc_out(f)   # (B, N0, num_out)
