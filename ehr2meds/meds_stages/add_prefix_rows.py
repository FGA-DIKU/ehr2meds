#!/usr/bin/env python
"""Custom MAP stage that creates a prefix column from composite codes."""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from ..utils import _changed, insert_parent_rows


@Stage.register(is_metadata=False)
def add_prefix_rows(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that creates a prefix column from composite codes and inserts prefix rows.

    The transform:
    1. Extracts the prefix (string before first '//') from the 'code' column.
    2. Removes the prefix from the 'code' column.
    3. Whenever the subject_id, time, or prefix changes, inserts a new row with 'code' set to the prefix.

    Args:
        stage_cfg: Configuration for the stage containing:
            - column: Column name to process (default: "code")

    Returns:
        Function that transforms a LazyFrame by splitting codes and inserting parent rows

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "time": [1, 1, 3],
        ...     "code": ["LAB//glucose", "LAB//sodium", "MEDICATION//aspirin"]
        ... }).lazy()
        >>> fn = add_prefix_rows(DictConfig({}))
        >>> result = fn(df).collect()
        >>> result.select("subject_id", "time", "code")
        shape: (5, 3)
        ┌────────────┬──────┬────────────┐
        │ subject_id ┆ time ┆ code       │
        │ ---        ┆ ---  ┆ ---        │
        │ i64        ┆ i64  ┆ str        │
        ╞════════════╪══════╪════════════╡
        │ 1          ┆ 1    ┆ LAB        │
        │ 1          ┆ 1    ┆ glucose    │
        │ 1          ┆ 1    ┆ sodium     │
        │ 2          ┆ 3    ┆ MEDICATION │
        │ 2          ┆ 3    ┆ aspirin    │
        └────────────┴──────┴────────────┘
    """

    header = stage_cfg.get("header", "")
    delimiter = stage_cfg.get("delimiter", "//")
    header_with_delimiter = f"{header}{delimiter}" if header else header

    def add_prefix_rows_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Process codes and insert prefix rows on changes."""
        parts = pl.col("code").str.splitn("//", 2)
        prefix = parts.struct.field("field_0")
        suffix = parts.struct.field("field_1")

        # 1. Extract prefix and cleaned code
        df = df.with_columns(
            prefix=pl.when(suffix.is_not_null()).then(pl.lit(header_with_delimiter) + prefix).otherwise(None),
            code=suffix.fill_null(pl.col("code")),
        )

        # 2. Detect when prefix or time or subject changes
        # We use prefix/time/subject_id change to trigger a new parent row
        change_cond = _changed("subject_id") | _changed("time") | _changed("prefix")
        trigger_cond = change_cond & pl.col("prefix").is_not_null()

        # 3. Use utility to insert parent rows
        return insert_parent_rows(
            df,
            change_cond=trigger_cond,
            code_expr=pl.col("prefix"),
        ).drop(["prefix"])

    return add_prefix_rows_fn
