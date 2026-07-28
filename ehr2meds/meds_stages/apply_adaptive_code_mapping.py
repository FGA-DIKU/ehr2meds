"""Apply a frozen adaptive code mapping to every MEDS data shard."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from ehr2meds.meds_stages.adaptive_code_mapping import apply_mapping, prepare_mapping
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


@Stage.register(
    is_metadata=False,
    default_config=Path("configs/MEDS/default_adaptive_code_mapping.yaml"),
)
def apply_adaptive_code_mapping_fntr(
    stage_cfg: DictConfig,
    code_metadata: pl.DataFrame,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the data transform from local and optional external mappings."""
    mapping = prepare_mapping(
        code_metadata,
        external_mapping_filepath=stage_cfg.get("mapping_filepath"),
        external_mapping_mode=str(stage_cfg.get("external_mapping_mode", "overlay")),
    )

    def transform(df: pl.LazyFrame) -> pl.LazyFrame:
        return apply_mapping(df, mapping)

    return transform


stage = apply_adaptive_code_mapping_fntr
