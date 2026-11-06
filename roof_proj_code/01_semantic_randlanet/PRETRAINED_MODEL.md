# Pretrained Columbus RandLA-Net model

## Download and verify

The reviewed checkpoint is now included at [`results/best.pth`](results/best.pth).
A normal Git clone includes it; Git LFS is not required.

[Download best.pth](https://github.com/Imp3rv1ous/columbus-lidar-lod2/raw/refs/heads/main/roof_proj_code/01_semantic_randlanet/results/best.pth)

- Size: 63,425,948 bytes (approximately 60.49 MiB).
- SHA-256: `8fa3a7b1c69369a4bb4185c875db30c1cefcb24a5183d717463083445e444830`.
- Contents: model state, optimizer state, scheduler state, epoch, selection score and RNG state.
- Architecture: the `RandLANet` implementation in this directory, 5,259,716 parameters,
  seven input channels (XYZ plus four features), four output classes.
- Stored epoch: 92 (zero-based; the 93rd epoch). The retained run completed 100 epochs.

From `roof_proj_code/01_semantic_randlanet/results`, verify the download with
`shasum -a 256 -c SHA256SUMS` on macOS, or `sha256sum -c SHA256SUMS` on Linux.

## Environment and paths

Install a PyTorch build appropriate for your hardware, then the other dependencies.
The retained training setup used an RTX 5090 and a CUDA 12.8-compatible PyTorch build;
a 5090 is not required just to load the model. CPU inference is supported but slower.

```bash
# Run from the repository root, after installing PyTorch.
python -m pip install -r roof_proj_code/01_semantic_randlanet/requirements.txt
cd roof_proj_code/01_semantic_randlanet
```

Default paths follow the published numbered directories:

- Source tiles: `roof_proj_code/00_LiDAR_data/`.
- Checkpoint: `roof_proj_code/01_semantic_randlanet/results/best.pth`.
- Generated preprocessing caches: `roof_proj_code/01_semantic_randlanet/cache/`.
- New predictions: `roof_proj_code/01_semantic_randlanet/outputs/preds/`.

`RANDLA_ROOT`, if set, must point to `roof_proj_code`, not the repository root
or the semantic-stage directory. Unset any old value when using automatic discovery.

CPU checkpoint-loading check (no point clouds required):

```bash
python - <<'PY'
from randla_net.config import Config
from randla_net.model.randla import RandLANet
from randla_net.engine.checkpoint import load_checkpoint
cfg = Config()
model = RandLANet(cfg, in_dim=3 + cfg.feat.in_dim, num_out=cfg.classes.num_classes)
checkpoint = load_checkpoint(cfg.results_dir / 'best.pth', model)
print('epoch:', checkpoint['epoch'], 'selection mIoU:', checkpoint['best'])
PY
```

## Source data and input assumptions

The experiment used 19 LAZ tiles from **USGS 3DEP, OH Columbus 2019 B19**, covering
Columbus's Near East Side: about 2.76 km² and 76,351,005 points (approximately
30 points/m² in the project notes). Obtain source point clouds through
[USGS LidarExplorer](https://apps.nationalmap.gov/lidar-explorer/) or
[The National Map Downloader](https://apps.nationalmap.gov/downloader/).
These are data access options, not a record of which download interface was used.

The current reader is dataset-specific:

- File naming: `USGS_LPC_OH_Columbus_2019_B19_<tile>.laz`.
- Source XYZ units: **US survey feet**, converted to meters in `prep/read_las.py`.
  The source CRS is NAD83(HARN) / Ohio South + NAVD88 (Geoid12B).
  Meter-valued or mixed-unit data require an explicit reader adjustment; do not
  apply the hard-coded feet conversion to such data.
- Required attributes: intensity, return number, number of returns and classification.
- ASPRS class-2 ground points provide the height-above-ground reference. The
  fallback for fewer than four ground points uses height above the tile minimum
  and does not reproduce a reliable local terrain reference.
- Preprocessing: 0.32 m voxel subsampling; 1 m terrain cells with four-neighbor
  inverse-distance interpolation for height above ground.
- Feature order: intensity normalized by the tile's 2nd/98th percentiles,
  return number / 7, number of returns / 7, and height above ground in meters.
  Keep this order and normalization when using the checkpoint.

For a Columbus-format input, place e.g.
`USGS_LPC_OH_Columbus_2019_B19_BS841708.laz` in `00_LiDAR_data`, then run:

```bash
python scripts/prep.py BS841708
python scripts/infer.py --tiles BS841708 --ckpt results/best.pth
# CPU-only option: append --device cpu to the inference command.
```

For all source tiles in that directory:

```bash
python scripts/prep.py all
python scripts/infer.py --all --ckpt results/best.pth
```

The output LAZ classification uses ASPRS 1=others, 2=ground, 5=vegetation,
6=building. Output point coordinates and the original header/CRS are retained.
Caches, original point clouds and the 19 locally retained prediction tiles are
not included in this repository; caches and new predictions are generated above.
Manual training labels are not required for inference.

## Training record and reproducibility limits

The four manually annotated training tiles were `BS842712`, `BS841710`,
`BS838711` and `BS843711`. Ground labels came from the source class-2 points;
manual building, vegetation and other annotations were mapped onto the subsampled
cloud by guarded nearest-neighbor matching. Unlabeled points were ignored by the loss.

The retained implementation uses Adam, inverse-square-root class-frequency-weighted
cross entropy, an initial learning rate of 0.01, exponential decay of 0.95 per epoch,
gradient clipping and CUDA mixed precision. Augmentation includes block centering,
Z-axis rotation, XY scaling and XY jitter. The training entry point defaults to
65,536 points per block, batch size 8 and 100 epochs. These are script defaults;
the checkpoint does not store the original full command line. The `--steps`
argument is the dataset's number of sampled blocks per epoch, not optimizer steps.

To retrain, supply your own annotations under
`00_LiDAR_data/training/<tile>/<tile>_{building,vegetation,others}.las`, update the
tile list in `randla_net/config.py` as needed, then run preprocessing,
`scripts/build_labels.py` and `scripts/train.py`. The original annotations are
local only, so the public repository alone does not reproduce the original training
run. `scripts/run_all.py` includes training; do not use it for inference alone.

The retained [`results/metrics.csv`](results/metrics.csv) records 100 epochs,
with about 42.9 minutes summed across the epoch training/evaluation timers.
This excludes setup, preprocessing, full-area inference and other experiments;
it is not the total GPU rental duration.

Best selection mIoU: 0.977269; building IoU: 0.978290. These were evaluated on
labeled points from the **training tiles** in evaluation mode, not a geographically
independent test set. They do not establish 1–2% accuracy for roof area, edge,
ridge or hip measurements. See [`results/RESULTS.md`](results/RESULTS.md) and the
[roof-plane practical notes](../04_05_roof_planes/PRACTICAL_NOTES.md).
