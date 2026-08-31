# RandLA-Net point-cloud segmentation

This directory contains the retained RandLA-Net implementation and model artifacts
for the Columbus airborne-LiDAR dataset.

## Retained result

- Checkpoint: `results/best.pth`
- Full-area predictions: `results/preds/` (19 LAZ tiles)
- Prediction classes: ASPRS 1=others, 2=ground, 5=vegetation, 6=building
- Prediction CRS: original compound CRS, including EPSG:3754 horizontal coordinates

The predictions are the intended starting point for new LoD2 research. Retraining
RandLA-Net is not required unless the segmentation itself becomes a research target.

## Code layout

- `randla_net/`: model, preprocessing, sampling, training and inference modules
- `scripts/`: command-line entry points and diagnostic checks
- `tests/`: deterministic unit tests for preprocessing and model operations
- `results/`: reviewed checkpoint, metrics and full-area predictions
- `outputs/`: disposable outputs created by a new local run
- `cache/`: disposable preprocessing cache created by `scripts/prep.py`

## Environment

RTX 5090 training requires a CUDA 12.8-compatible PyTorch build. Install PyTorch
first, then install `requirements.txt`.

```bash
export RANDLA_ROOT=/path/to/roof_proj_code
cd "$RANDLA_ROOT/randla-net"
```

The project root is detected automatically when `RANDLA_ROOT` is not set.

## Verification

```bash
python tests/test_prep.py
python tests/test_model.py
```

The retained checkpoint can be loaded on CPU without running inference:

```bash
python -c "from randla_net.config import Config; from randla_net.model.randla import RandLANet; from randla_net.engine.checkpoint import load_checkpoint; c=Config(); m=RandLANet(c, 3+c.feat.in_dim, 4); print(load_checkpoint(c.results_dir/'best.pth', m)['best'])"
```

## Inference

Inference requires preprocessing caches. Generate them and write new predictions
under `outputs/`; the reviewed files under `results/` are not overwritten.

```bash
python scripts/prep.py all
python scripts/infer.py --all
```

## Training

The reproducible full workflow is:

```bash
python scripts/run_all.py
```

This command preprocesses data, builds labels, runs tests, trains, and performs a
two-tile diagnostic inference. `--all-tiles` still includes training; do not use it
when the goal is inference only.

Reported IoU values were measured on labeled points from the training tiles and
were used for checkpoint selection. They are not an independent test-set estimate
of geographic generalization. See `results/RESULTS.md`.

## 论文依据（步骤 1 · 语义分割）

- 方法：**Hu, Q., Yang, B., Xie, L., Rosa, S., Guo, Y., Wang, Z., Trigoni, N., Markham, A.
  (2020).** RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds.
  *CVPR 2020*.
