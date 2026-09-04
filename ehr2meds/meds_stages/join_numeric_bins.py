"""Join valid numeric bin indices onto laboratory codes."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def join_numeric_bins(data: pl.LazyFrame, bin_column: str, separator: str) -> pl.LazyFrame:
    """Rewrite binned codes as ``<code><separator><bin>``.

    Rows without a valid numeric bin retain their original code. All other
    event columns, including the raw and derived numeric values, are unchanged.
    """
    joined_code = pl.concat_str(
        pl.col(DataSchema.code_name),
        pl.lit(separator),
        pl.col(bin_column).cast(pl.String),
    )

    has_bin = pl.col(bin_column).is_not_null()
    code = pl.when(has_bin).then(joined_code).otherwise(pl.col(DataSchema.code_name))
    return data.with_columns(**{DataSchema.code_name: code})


def join_numeric_bins_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the final code-representation transform."""
    bin_column = stage_cfg.numeric_value_columns.bin_index
    separator = str(stage_cfg.separator)

    def transform(df: pl.LazyFrame) -> pl.LazyFrame:
        return join_numeric_bins(df, bin_column=bin_column, separator=separator)

    return transform


stage = Stage.register(
    is_metadata=False,
    default_config=Path("configs/MEDS/default_numeric_values.yaml"),
)(join_numeric_bins_fntr)
