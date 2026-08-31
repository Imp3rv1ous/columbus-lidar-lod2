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

**Study area.** Near East Side, Columbus, Ohio, covered by USGS 3DEP 2019 airborne
LiDAR tiles.

## Progress so far

The pipeline currently runs from raw tiles through to regularized building outlines.

Points across the study area are classified into ground, vegetation, building and
other returns. Building points are then separated into individual structures,
including neighbouring buildings whose footprints touch in plan view. Non-building
objects that survive classification — bridge decks, highway ramps, tree rows and
power lines — are identified and removed, with a recorded reason for each removal.

For every building, the roof is decomposed into planar faces, including small
features such as dormers and narrow projections. Ridge, valley and step lines are
derived analytically from the geometry of adjacent planes, and the ragged raster
boundary of each building is regularized into a simplified polygon that follows the
building's own dominant orientation rather than an assumed global grid.

Assembly of these elements into watertight LOD2 solids, semantic export to CityJSON,
and quantitative accuracy evaluation are planned but not yet implemented.

## Repository contents

Source code, method documentation and result tables only. Directories are numbered
by pipeline stage, and each one has its own README describing the method and the
literature it follows; `roof_proj_code/PIPELINE.md` is the overall index.

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
