"""Apply a frozen adaptive code mapping to every MEDS data shard."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from ehr2meds.adaptive_code_mapping import MAPPED_CODE_COLUMN, prepare_mapping
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def apply_mapping(data: pl.LazyFrame, mapping: pl.DataFrame) -> pl.LazyFrame:
    """Rewrite codes through a frozen mapping while preserving row order and schema."""
    lookup = mapping.select(DataSchema.code_name, MAPPED_CODE_COLUMN).lazy()
    return (
        data.join(lookup, on=DataSchema.code_name, how="left", maintain_order="left")
        .with_columns(pl.coalesce(MAPPED_CODE_COLUMN, DataSchema.code_name).alias(DataSchema.code_name))
        .drop(MAPPED_CODE_COLUMN)
    )


@Stage.register(
    is_metadata=False,
    default_config=Path("configs/MEDS/default_adaptive_code_mapping.yaml"),
)
def apply_adaptive_code_mapping_fntr(
    stage_cfg: DictConfig,
    code_metadata: pl.DataFrame,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the data transform from the local fitted mapping or an external one."""
    mapping = prepare_mapping(code_metadata, external_mapping_filepath=stage_cfg.get("mapping_filepath"))

    def transform(df: pl.LazyFrame) -> pl.LazyFrame:
        return apply_mapping(df, mapping)

    return transform


stage = apply_adaptive_code_mapping_fntr
