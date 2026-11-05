#!/usr/bin/env python
"""Custom MAP stage that adds a row with HH:MM when subject or minute-level time changes."""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from ..utils import _changed, ensure_datetime_column, insert_parent_rows

END_OF_DAY = pl.time(23, 59, 59, 999999)


@Stage.register(is_metadata=False)
def add_time_rows(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that inserts a row with HHh MMm whenever the subject_id changes,
    or the timestamp changes at minute precision.

    Args:
        stage_cfg: Configuration for the stage containing:
            - header: Optional string to prepend to the HH:MM time (default: "")

    Returns:
        Function that transforms a LazyFrame by inserting time rows

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2],
        ...     "time": [
        ...         datetime(2024, 1, 1, 10, 0),
        ...         datetime(2024, 1, 1, 10, 0),
        ...         datetime(2024, 1, 1, 10, 5),
        ...         datetime(2024, 1, 1, 10, 5),
        ...     ],
        ...     "code": ["A", "B", "C", "D"]
        ... }).lazy()
        >>> fn = add_time_rows(DictConfig({"header": "TIME//"}))
        >>> result = fn(df).collect()
        >>> result.select("subject_id", "time", "code")
        shape: (7, 3)
        ┌────────────┬─────────────────────┬────────────┐
        │ subject_id ┆ time                ┆ code       │
        │ ---        ┆ ---                 ┆ ---        │
        │ i64        ┆ datetime[μs]        ┆ str        │
        ╞════════════╪═════════════════════╪════════════╡
        │ 1          ┆ 2024-01-01 10:00:00 ┆ TIME//10:00│
        │ 1          ┆ 2024-01-01 10:00:00 ┆ A          │
        │ 1          ┆ 2024-01-01 10:00:00 ┆ B          │
        │ 1          ┆ 2024-01-01 10:05:00 ┆ TIME//10:05│
        │ 1          ┆ 2024-01-01 10:05:00 ┆ C          │
        │ 2          ┆ 2024-01-01 10:05:00 ┆ TIME//10:05│
        │ 2          ┆ 2024-01-01 10:05:00 ┆ D          │
        └────────────┴─────────────────────┴────────────┘
    """

    header = stage_cfg.get("header", "")
    delimiter = stage_cfg.get("delimiter", "//")
    header_with_delimiter = f"{header}{delimiter}" if header else header

    eod_string = stage_cfg.get("eod_string", None)

    def add_time_rows_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Insert HHh MMm rows on subject changes or minute-level time changes."""
        df = ensure_datetime_column(df, "time")

        # 1. Expressions for HHh MMm parsing and checking
        time_hh_mm = pl.col("time").dt.strftime("%Hh//%Mm")
        time_minute = pl.col("time").dt.truncate("1m")
        has_time = pl.col("time").is_not_null()

        # 2. Detect when subject changes or the timestamp changes at minute precision.
        # This treats any change in (date, hour, minute) as a new time block, while
        # ignoring changes below the minute (seconds, microseconds). Skip static
        # (null-time) events — they have no clock time to announce.
        change_cond = (_changed("subject_id") | _changed(time_minute)) & has_time

        # Build the new code column expression
        code_expr = pl.lit(header_with_delimiter) + time_hh_mm
        if eod_string is not None:
            is_eod = pl.col("time").dt.time() == END_OF_DAY
            code_expr = pl.when(is_eod).then(pl.lit(header_with_delimiter + eod_string)).otherwise(code_expr)
            # Differentiates between 23:59 and 23:59:59.999999
            change_cond = change_cond | (_changed(is_eod) & has_time)

        # 3. Use utility to insert parent rows
        return insert_parent_rows(
            df,
            change_cond=change_cond,
            code_expr=code_expr,
        )

    return add_time_rows_fn
