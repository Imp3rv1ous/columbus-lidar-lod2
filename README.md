# LiDAR Building Reconstruction Pipeline

Part of an ongoing project on **extracting building information from historical
Sanborn fire insurance maps and LiDAR data using machine learning and computer
vision methods**. In that project, Sanborn maps provide the primary visual evidence
for building detection, while airborne LiDAR supplies the geometric context used to
validate and refine the extracted features.

This repository holds the LiDAR side of the work: a pipeline that turns raw airborne
LiDAR point clouds into structured building geometry — semantic segmentation,
per-building instance separation, roof plane extraction, and regularized building
outlines.

**Study area.** Near East Side, Columbus, Ohio — 2.76 km², 19 USGS 3DEP 2019 tiles,
76,351,005 points at roughly 30 pts/m².

## Pipeline

| Step | Stage | Status | Directory |
|---|---|---|---|
| 0 | Raw LiDAR data | done | `roof_proj_code/00_LiDAR_data/` |
| 1 | Semantic segmentation (RandLA-Net) | done | `roof_proj_code/01_semantic_randlanet/` |
| 2 | Building instance segmentation | done | `roof_proj_code/02_03_instance_seg/` |
| 3 | Non-building removal (Gate A) | done | same |
| 4 | Roof plane extraction | done | `roof_proj_code/04_05_roof_planes/` |
| 5 | Structural cue layer (ridge / valley / step) | done | same |
| 6 | Outline regularization | done | `roof_proj_code/06_outline_reg/` |
| 7 | Model assembly (LOD2 geometry) | planned | — |
| 8 | Semantic export (CityJSON) | planned | — |
| 9 | Accuracy evaluation | planned | — |

## Results so far

- Semantic segmentation: mIoU 0.977 (checkpoint-selection metric, not an independent
  geographic test set); 8.6M building points (11.3% of the cloud)
- Instance segmentation: 2,296 building instances
- After Gate A cleaning: 2,267 clean instances
- Roof planes: 33,845 planes (26,643 roof + 7,202 wall), 91.2% point assignment rate
- Structural cues: 8,830 ridges, 3,963 valleys, 14,111 step edges
- Outline regularization: 2,253 / 2,261 successful (99.65%), median 7 vertices,
  median offset 0.15 m

## Repository contents

Source code, method documentation and result tables only. Each pipeline step has its
own README describing the method and its references; `roof_proj_code/PIPELINE.md` is
the overall index.

Not included, for size and licensing reasons:

- Point clouds (`.laz` / `.las` / `.ply`, ~1.9 GB) — the source data is publicly
  available from the [USGS 3D Elevation Program](https://www.usgs.gov/3d-elevation-program)
- Model weights (`results/best.pth`, 60 MB)
- Third-party reference code and papers — please obtain these from their original sources

## Environment

Dependencies are listed in `roof_proj_code/01_semantic_randlanet/requirements.txt`.
Point cloud processing relies mainly on `laspy`, `numpy`, `scipy`, `scikit-image` and
`shapely`; the segmentation network uses `torch`.

## Data source

USGS 3D Elevation Program (3DEP), Columbus 2019 B19.
CRS: NAD83(HARN) / Ohio South (ftUS) + NAVD88 (Geoid12B).

## People

- Student: Ochila Bayuud, Department of Electrical and Computer Engineering
- Advisor: Prof. Irem Eryilmaz, Department of Electrical and Computer Engineering
- Co-Advisor: Prof. Ningchuan Xiao, Department of Geography
