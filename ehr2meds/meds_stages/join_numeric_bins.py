"""Join valid numeric bin indices onto laboratory codes."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

CODE = DataSchema.code_name
NUMERIC_BIN = "numeric_value_bin"


def join_numeric_bins(data: pl.LazyFrame, *, separator: str = "//") -> pl.LazyFrame:
    """Rewrite binned codes as ``<code><separator><bin>``.

    Rows without a valid numeric bin retain their original code. All other
    event columns, including the raw and derived numeric values, are unchanged.
    """
    joined_code = pl.concat_str(
        pl.col(CODE),
        pl.lit(separator),
        pl.col(NUMERIC_BIN).cast(pl.String),
    )

    return data.with_columns(pl.when(pl.col(NUMERIC_BIN).is_not_null()).then(joined_code).otherwise(pl.col(CODE)).alias(CODE))


def join_numeric_bins_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the final code-representation transform."""
    separator = stage_cfg.get("separator", "//")

    def transform(df: pl.LazyFrame) -> pl.LazyFrame:
        return join_numeric_bins(df, separator=separator)

    return transform


stage = Stage.register(is_metadata=False)(join_numeric_bins_fntr)
