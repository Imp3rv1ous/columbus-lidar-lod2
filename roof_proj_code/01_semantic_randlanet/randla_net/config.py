"""Single source of truth for paths, class scheme and the feature recipe.

Pure dataclasses (no external config dep yet); YAML/OmegaConf can wrap this later.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

# US survey foot -> meter (the Columbus tiles are in NAD83 Ohio South State Plane, ftUS)
FT2M: float = 0.3048006096012192

# Training tiles for the retained RandLA-Net model.
_TRAIN_TILES = ("BS842712", "BS841710", "BS838711", "BS843711")


@dataclass(frozen=True)
class ClassScheme:
    """Internal training codes (0 is the ignored / unlabeled bin)."""

    ignore_index: int = 0
    # internal id -> human name
    names: tuple[tuple[int, str], ...] = (
        (0, "unlabeled"),
        (1, "others"),
        (2, "ground"),
        (3, "vegetation"),
        (4, "building"),
    )
    # internal id -> ASPRS code (used only when writing predictions back to .las)
    to_asprs: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (3, 5), (4, 6))

    @property
    def num_classes(self) -> int:
        # trained classes exclude the ignored bin
        return len(self.names) - 1

    def name(self, code: int) -> str:
        return dict(self.names)[code]


@dataclass(frozen=True)
class FeatureRecipe:
    """Ordered feature channels appended to xyz. Change here = change everywhere."""

    channels: tuple[str, ...] = (
        "intensity_norm",
        "return_number_norm",
        "num_returns_norm",
        "hag",
    )

    @property
    def in_dim(self) -> int:
        return len(self.channels)


@dataclass(frozen=True)
class Config:
    # override on another machine with:  export RANDLA_ROOT=/path/to/roof_proj_code
    root: Path = field(
        default_factory=lambda: Path(os.environ.get("RANDLA_ROOT", _DEFAULT_ROOT)).expanduser().resolve()
    )

    sub_grid: float = 0.32  # voxel size for grid subsampling, METERS
    hag_cell: float = 1.0   # DTM cell size for height-above-ground, METERS
    hag_knn: int = 4        # inverse-distance neighbours for DTM lookup

    # --- block sampling / network shape (RandLA-Net lineage) ---
    num_points: int = 65536               # points per training block
    k_n: int = 16                         # KNN neighbours per layer
    num_layers: int = 5
    sub_ratios: tuple[int, ...] = (4, 4, 4, 4, 2)   # per-layer downsample factor
    d_out: tuple[int, ...] = (16, 64, 128, 256, 512)
    batch_size: int = 4
    noise_init: float = 3.5               # center-jitter scale for the sampler

    # --- training ---
    learning_rate: float = 1e-2
    lr_decay: float = 0.95                # per-epoch ExponentialLR
    grad_clip: float = 1.0
    steps_per_epoch: int = 500
    max_epochs: int = 100

    classes: ClassScheme = field(default_factory=ClassScheme)
    feat: FeatureRecipe = field(default_factory=FeatureRecipe)

    train_tiles: tuple[str, ...] = _TRAIN_TILES
    val_tile: str = "BS841708"

    # --- derived paths ---
    @property
    def data_dir(self) -> Path:
        return self.root / "LiDAR_data"

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "training"

    @property
    def cache_dir(self) -> Path:
        return self.root / "randla-net" / "cache"

    @property
    def out_dir(self) -> Path:
        return self.root / "randla-net" / "outputs"

    @property
    def results_dir(self) -> Path:
        """Reviewed model artifacts synced back from the training machine."""
        return self.root / "randla-net" / "results"

    def orig_laz(self, tile: str) -> Path:
        """Return the canonical original full-tile LAZ path."""
        return self.data_dir / f"USGS_LPC_OH_Columbus_2019_B19_{tile}.laz"
