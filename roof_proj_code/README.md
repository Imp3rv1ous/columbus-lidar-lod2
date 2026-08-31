# Columbus LiDAR: RandLA-Net to LoD2 research workspace

## Current starting point

The previous LoD2 pipeline and its outputs have been removed. New LoD2 research
starts from the retained RandLA-Net classified point clouds:

`01_semantic_randlanet/results/preds/`

The directory contains 19 LAZ tiles and 76,351,005 points. Every tile preserves
the original CRS and uses ASPRS classes 1 (others), 2 (ground), 5 (vegetation),
and 6 (building).

## Retained project material

- `00_LiDAR_data/`: one canonical copy of each original USGS tile plus retained
  training annotations
- `01_semantic_randlanet/`: verified segmentation code, final checkpoint and predictions
- `ref/`: papers relevant to RandLA-Net and new LoD2 research
- `nsf-hegs-2025-main-proposal.pdf`: project-level proposal context

## Scope of the next research stage

No LoD2 method has been selected yet. The next discussion should independently
decide:

1. how building instances and boundaries should be derived from classified points;
2. how roof planes and topology should be reconstructed;
3. which LoD2 representation and exchange format are required;
4. how segmentation, instance, geometric and topology quality will be evaluated.

Old LoD2 conclusions are intentionally not treated as assumptions in this new stage.
