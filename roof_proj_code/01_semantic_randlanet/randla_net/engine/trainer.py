"""Custom training loop: AMP (CUDA only), grad clip, LR decay, eval, checkpointing.

Transparent ~loop (no Lightning) so the masked-loss + multiscale path stays
debuggable. Device-agnostic: CPU/MPS dev on Mac, CUDA on the 5090.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from ..config import Config
from ..losses.weighted_ce import masked_weighted_ce
from .evaluator import ConfusionMatrix
from .checkpoint import save_checkpoint


class Trainer:
    def __init__(self, cfg: Config, model, train_loader, device: str,
                 weight=None, val_loader=None, out_dir: Path | None = None,
                 num_classes: int = 4, logger=None):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weight = weight.to(device) if weight is not None else None
        self.num_classes = num_classes
        self.logger = logger
        # output index 0..C-1 == internal class 1..C (others/ground/veg/building)
        self.class_names = [cfg.classes.name(i + 1) for i in range(num_classes)]
        self.out_dir = Path(out_dir) if out_dir else Path("outputs/run")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=cfg.lr_decay)
        self.use_amp = self.dev_type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.best = 0.0
        self._csv = self.out_dir / "metrics.csv"
        with open(self._csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_miou", "val_miou", "lr", "secs"]
                                   + [f"IoU_{n}" for n in self.class_names])

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        cm = ConfusionMatrix(self.num_classes)
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=self.dev_type, enabled=self.use_amp):
                    logits = self.model(batch)
                    loss = masked_weighted_ce(logits, batch.labels, self.weight)
                if train:
                    self.opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    if self.use_amp:
                        self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
            cm.update(logits.detach().float(), batch.labels)
            total += float(loss.item())
            n += 1
        return total / max(n, 1), cm

    def fit(self, epochs: int):
        log = self.logger if (self.logger is not None and self.logger.enabled) else None
        if log:
            log.header("Training (per-epoch)")
            log.line("`train_mIoU` is in **train mode** (BatchNorm batch stats); "
                     "`eval_mIoU` and the per-class IoU are in **eval mode** "
                     "(running stats) — the honest, inference-matching numbers. "
                     "best.pth is selected on `eval_mIoU`. Watch the **building** column.\n")
            log.table_header(["ep", "loss", "train_mIoU", "eval_mIoU"]
                             + [f"IoU_{n[:4]}" for n in self.class_names]
                             + ["lr", "secs", "best?"])

        for ep in range(epochs):
            t0 = time.time()
            tr_loss, tr_cm = self._run_epoch(self.train_loader, train=True)
            self.sched.step()
            val_miou = float("nan")
            eval_cm = tr_cm
            if self.val_loader is not None:
                _, va_cm = self._run_epoch(self.val_loader, train=False)
                val_miou = va_cm.miou()
                eval_cm = va_cm
            lr = self.opt.param_groups[0]["lr"]
            secs = time.time() - t0
            per_class = eval_cm.iou().tolist()
            print(f"epoch {ep:3d} | loss {tr_loss:.4f} | train_mIoU {tr_cm.miou():.3f} "
                  f"| eval_mIoU {val_miou:.3f} | IoU(bld) {per_class[-1]:.3f} "
                  f"| lr {lr:.4f} | {secs:.1f}s")
            with open(self._csv, "a", newline="") as f:
                csv.writer(f).writerow([ep, tr_loss, tr_cm.miou(), val_miou, lr, secs] + per_class)

            score = val_miou if self.val_loader is not None else tr_cm.miou()
            is_best = score >= self.best
            save_checkpoint(self.out_dir / "last.pth", self.model, self.opt, self.sched, ep, self.best)
            if is_best:
                self.best = score
                save_checkpoint(self.out_dir / "best.pth", self.model, self.opt, self.sched, ep, self.best)

            if log:
                log.row([ep, f"{tr_loss:.4f}", f"{tr_cm.miou():.3f}", f"{val_miou:.3f}"]
                        + [f"{v:.3f}" for v in per_class]
                        + [f"{lr:.4f}", f"{secs:.0f}", "✓" if is_best else ""])

        if log:
            log.line(f"\n**best eval_mIoU = {self.best:.4f}** (selection metric)\n")
        return self.best
