"""Apply a frozen adaptive code mapping to every MEDS data shard."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from ehr2meds.adaptive_code_mapping import CODE_COLUMN, MAPPED_CODE_COLUMN, prepare_mapping
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def apply_mapping(data: pl.LazyFrame, mapping: pl.DataFrame) -> pl.LazyFrame:
    """Rewrite codes through a frozen mapping while preserving row order and schema."""
    temporary = "_adaptive_mapped_code"
    lookup = mapping.select(CODE_COLUMN, pl.col(MAPPED_CODE_COLUMN).alias(temporary)).lazy()
    return (
        data.join(lookup, on=CODE_COLUMN, how="left", maintain_order="left")
        .with_columns(pl.coalesce(temporary, CODE_COLUMN).alias(CODE_COLUMN))
        .drop(temporary)
    )


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
