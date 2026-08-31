"""One-command clean run: prep -> labels -> unit tests -> train -> infer.

Every stage appends to ONE markdown report (RANDLA_REPORT) so after the run the user
syncs back a single reviewable file (per-epoch per-class IoU + inference class
distributions + warnings). Aborts on the first failing stage.

  cd /workspace/proj/randla-net
  export RANDLA_ROOT=/workspace/proj
  export RANDLA_REPORT=outputs/run_report.md
  python3 scripts/run_all.py                 # GPU, 32 workers, batch 8, 100 epochs
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from randla_net.config import Config  # noqa: E402
from randla_net.common.runlog import RunLogger  # noqa: E402

PY = sys.executable


def run(cmd, log, title):
    print(f"\n########## {title}: {' '.join(cmd)}")
    log.header(title)
    log.line(f"`{' '.join(cmd)}`\n")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), env=os.environ)
    secs = time.time() - t0
    log.line(f"\n→ exit={r.returncode}  ({secs:.0f}s)\n")
    if r.returncode != 0:
        log.warn(f"**{title} FAILED (exit {r.returncode}) — run aborted.**")
        print(f"!!!!!!!!!! {title} FAILED (exit {r.returncode}); aborting")
        sys.exit(r.returncode)
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None, help="default outputs/")
    ap.add_argument("--preds", default=None, help="default outputs/preds/")
    ap.add_argument("--all-tiles", action="store_true",
                    help="after the diagnostic run, prep + infer ALL 19 tiles (full dataset)")
    ap.add_argument("--holdout-tile", default=None,
                    help="pass through to train.py: hold one labeled train tile out as a TRUE "
                         "unseen-tile validation so best.pth is selected on generalization "
                         "(default None = select on the train-tile eval metric)")
    args = ap.parse_args()

    cfg = Config()
    args.out = args.out or str(cfg.out_dir)
    args.preds = args.preds or str(cfg.out_dir / "preds")
    report = os.environ.get("RANDLA_REPORT", "outputs/run_report.md")
    if not os.path.isabs(report):
        report = str((ROOT / report).resolve())
    os.environ["RANDLA_REPORT"] = report       # children inherit the absolute path
    Path(report).parent.mkdir(parents=True, exist_ok=True)

    with open(report, "w") as f:                # fresh report
        f.write("# RandLA-Net clean run report\n")
    log = RunLogger(report)

    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda_ok = bool(torch.cuda.is_available())
        tver = torch.__version__
    except Exception as e:                       # pragma: no cover
        gpu, cuda_ok, tver = f"(torch import failed: {e})", False, "?"

    log.header("Environment")
    log.kv({
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": tver,
        "cuda_available": cuda_ok,
        "gpu": gpu,
        "RANDLA_ROOT": os.environ.get("RANDLA_ROOT", "(unset)"),
        "data_dir": cfg.data_dir,
        "epochs": args.epochs, "workers": args.workers, "batch_size": args.batch_size,
        "out": args.out, "preds": args.preds,
    })
    if not cuda_ok:
        log.warn("CUDA not available — training will be very slow on CPU.")

    prep_tiles = list(cfg.train_tiles) + [cfg.val_tile]

    t_start = time.time()
    run([PY, "scripts/prep.py", *prep_tiles], log, "1. Prep")
    run([PY, "scripts/build_labels.py"], log, "2. Build labels")
    run([PY, "tests/test_prep.py"], log, "3a. Unit tests: prep")
    run([PY, "tests/test_model.py"], log, "3b. Unit tests: model")
    train_cmd = [PY, "scripts/train.py", "--epochs", str(args.epochs), "--device", "cuda",
                 "--workers", str(args.workers), "--batch-size", str(args.batch_size),
                 "--out", args.out]
    if args.holdout_tile:
        train_cmd += ["--holdout-tile", args.holdout_tile]
    sel = f"holdout {args.holdout_tile} (generalization)" if args.holdout_tile \
        else "train-tile eval (selection, not generalization)"
    run(train_cmd, log, f"4. Train [best.pth selected on: {sel}]")
    run([PY, "scripts/infer.py", "--ckpt", f"{args.out}/best.pth",
         "--tiles", "BS842712", cfg.val_tile, "--out", args.preds],
        log, "5. Infer (diagnostic: BS842712 + val tile)")

    if args.all_tiles:
        run([PY, "scripts/prep.py", "all"], log, "6. Prep all 19 tiles")
        run([PY, "scripts/infer.py", "--ckpt", f"{args.out}/best.pth", "--all",
             "--out", args.preds], log, "7. Infer all 19 tiles (full dataset)")

    log.header("Done")
    log.kv({"total_minutes": f"{(time.time() - t_start) / 60:.1f}",
            "checkpoints": args.out, "predictions": args.preds})
    log.line("Sync back `outputs/` (report + preds + best.pth). Review the **building** "
             "column in Training and the **building%** in Inference "
             "(BS842712 is a training tile — building% there should be clearly > 0).")
    print(f"\n########## ALL DONE in {(time.time() - t_start) / 60:.1f} min. Report: {report}")


if __name__ == "__main__":
    main()
