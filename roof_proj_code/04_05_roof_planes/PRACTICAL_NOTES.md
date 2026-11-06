# Practical roof-plane notes

These notes summarize the retained code and experiment records. They describe
observed failure modes and implementation limits, not validated estimating accuracy.

## Semantic segmentation is only the first stage

The RandLA-Net checkpoint classifies ground, vegetation, buildings and other points.
It does not predict roof facets, hips or takeoff dimensions. Building instance
separation and geometric plane extraction run downstream. The pipeline includes
manual review of some candidate non-building instances.

## Edges, narrow facets and complex roofs

- **Cross-facet normals:** a 20-neighbor PCA neighborhood can span multiple facets
  on dormers and narrow projections. The main region-growing pass can therefore
  miss these faces. The retained extractor adds a smaller-neighborhood pass and
  a third pass that fits trimmed planes to connected residual points without
  relying on normals. See `plane_extraction.py` and the stage README.
- **Preserving steps:** merging nearly parallel planes too aggressively can erase
  real height discontinuities. The big-plane merge includes a 0.10 m offset gate;
  small-into-large absorption also checks the joint fit RMS. These thresholds
  were tuned on this dataset, rather than validated across arbitrary densities.
- **Disconnected RANSAC support:** the recorded 15-building comparison found
  disconnected "phantom planes" in the sequential-RANSAC route. Spatial coherence
  is part of why the retained approach uses region growing and connected support.
  See `output_pilot/rg_vs_jung_stats.csv`.
- **Vegetation filtering at eaves:** the plane-level neighborhood rule was rolled
  back because its expanded neighborhood reached beyond eaves and removed real
  roof-edge faces near trees. See the vegetation section of the stage README.
- **Return-number filtering is not a simple fix:** the upstream experiment removing
  all intermediate returns removed 9.3% of the points from a clean gable-roof
  example, concentrated at its eaves, while removing only 16% of the tested tree
  contamination. This prefilter was rejected. See the experiment notes in
  `../02_03_instance_seg/README.md`.
- **Inference point ordering matters:** the semantic inference code shuffles block
  points before multiscale subsampling, matching training. Keeping the KD-tree's
  distance-sorted order instead caused coarse layers to retain block centers and
  lose edge/clustered building evidence. See `../01_semantic_randlanet/randla_net/engine/inference.py`.

## Measurement limits

`full_run.py` computes the table's plane area from occupied **0.5 m XY grid cells**.
That is an approximate horizontal projected area, not the true sloped surface area
of a bounded roof facet. It should not be treated as a finished roofing takeoff.

The published cue-layer scripts produce geometric ridge/valley/step evidence and
rough outlines. Their presence does not establish correct hip endpoints, complete
facet boundaries or watertight solids. Accurate takeoffs still require validated
boundaries and topology, plus a suitable measurement reference. Report total roof
area, individual facet area and edge/ridge/hip length errors separately.

The high semantic IoU in the training results is a training-tile selection metric.
Neither it nor point-to-plane RMS demonstrates 1–2% takeoff accuracy.

## Running downstream scripts elsewhere

Several downstream scripts retain machine-specific `BASE` and input/output paths;
adapt these before using them on another machine. Their source point-cloud units
also assume US survey feet, with internal geometric calculations in meters.
Some cue/outline exports are in ftUS; consult each script rather than assuming all
exported geometry shares the same unit.

The portable checkpoint/inference path is documented in
[`../01_semantic_randlanet/PRETRAINED_MODEL.md`](../01_semantic_randlanet/PRETRAINED_MODEL.md).
