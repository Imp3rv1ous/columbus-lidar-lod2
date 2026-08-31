"""Ignore-aware streaming confusion matrix -> per-class IoU / mIoU / OA."""
from __future__ import annotations

import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.C = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0) -> None:
        lb = labels.reshape(-1)
        valid = lb != ignore_index
        if valid.sum() == 0:
            return
        pred = logits.reshape(-1, self.C).argmax(-1)[valid]   # 0..C-1
        tgt = lb[valid] - 1                                   # 1..C -> 0..C-1
        k = (tgt * self.C + pred).cpu()
        self.mat += torch.bincount(k, minlength=self.C * self.C).reshape(self.C, self.C)

    def iou(self) -> torch.Tensor:
        m = self.mat.double()
        tp = m.diag()
        denom = m.sum(0) + m.sum(1) - tp
        return tp / (denom + 1e-9)

    def miou(self) -> float:
        # average only over classes present (in pred or target): a class absent
        # from this eval sample must not score a hard 0 and drag selection around.
        iou = self.iou()
        present = (self.mat.sum(0) + self.mat.sum(1)) > 0
        if int(present.sum()) == 0:
            return 0.0
        return iou[present].mean().item()

    def oa(self) -> float:
        return (self.mat.diag().sum().double() / (self.mat.sum().double() + 1e-9)).item()
