# LiDAR Building Reconstruction Pipeline

Part of an ongoing project on **extracting building information from historical
Sanborn fire insurance maps and LiDAR data using machine learning and computer
vision methods**. Sanborn maps provide the visual evidence for building detection;
airborne LiDAR supplies the geometric context used to validate and refine the
extracted features.

This repository holds the LiDAR side of that work — a pipeline that turns raw
airborne LiDAR point clouds into structured building geometry. Study area: the Near
East Side of Columbus, Ohio, covered by USGS 3DEP 2019 tiles.

## Progress so far

- **Semantic segmentation** — points classified into ground, vegetation, building
  and other returns
- **Instance segmentation** — building points separated into individual structures,
  including neighbouring buildings whose footprints touch in plan view
- **Non-building removal** — bridge decks, highway ramps, tree rows and power lines
  identified and removed, each with a recorded reason
- **Roof plane extraction** — every roof decomposed into planar faces, down to
  dormers and narrow projections
- **Structural cue layer** — ridge, valley and step lines derived analytically from
  the geometry of adjacent planes
- **Outline regularization** — ragged raster boundaries turned into simplified
  polygons that follow each building's own dominant orientation

Not yet implemented: assembly into watertight LOD2 solids, CityJSON export, and
quantitative accuracy evaluation.

## Repository contents

Source code, method documentation and result tables only. Directories are numbered
by pipeline stage, and each has its own README describing the method and the
literature it follows; `roof_proj_code/PIPELINE.md` is the overall index.

Not included, for size and licensing reasons:

- Point clouds (`.laz` / `.las` / `.ply`) — the source data is publicly available
  from the [USGS 3D Elevation Program](https://www.usgs.gov/3d-elevation-program)
- Trained model weights
- Third-party reference code and papers — please obtain these from their original
  sources

## Environment

Dependencies are listed in `roof_proj_code/01_semantic_randlanet/requirements.txt`.
Point cloud processing relies mainly on `laspy`, `numpy`, `scipy`, `scikit-image`
and `shapely`; the segmentation network uses `torch`.

## Data source

USGS 3D Elevation Program (3DEP), Columbus 2019 B19.
CRS: NAD83(HARN) / Ohio South (ftUS) + NAVD88 (Geoid12B).
