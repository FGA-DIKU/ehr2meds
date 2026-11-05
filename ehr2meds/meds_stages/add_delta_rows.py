#!/usr/bin/env python
"""Custom MAP stage that adds Delta rows when subjects or dates change."""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from ..utils import ensure_datetime_column


@Stage.register(is_metadata=False)
def add_delta_rows(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that inserts rows like "Delta +N days" on changes.

    The row is inserted whenever the subject_id or the date of the event changes.
    N is computed as the date difference from the previous row for the same subject.
    If N exceeds ``max_delta_days``, multiple rows are inserted whose deltas sum to N.
    Does not insert a Delta row for the first event of each subject (i.e. no "Delta +0 days" rows).

    Args:
        stage_cfg: Configuration for the stage containing:
            - header: Prefix for generated code rows (default: "Delta")
            - max_delta_days: Maximum days per generated row (default: 365)

    Returns:
        Function that transforms a LazyFrame by inserting age rows

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1],
        ...     "time": [
        ...         datetime(2020, 1, 1),
        ...         datetime(2021, 1, 5),
        ...         datetime(2021, 1, 6),
        ...     ],
        ...     "code": ["A", "B", "C"],
        ... }).lazy()
        >>> fn = add_delta_rows(DictConfig({"max_delta_days": 365}))
        >>> result = fn(df).collect()
        >>> result.select("subject_id", "time", "code")
        shape: (7, 3)
        ┌────────────┬─────────────────────┬───────────────────┐
        │ subject_id ┆ time                ┆ code              │
        │ ---        ┆ ---                 ┆ ---               │
        │ i64        ┆ datetime[μs]        ┆ str               │
        ╞════════════╪═════════════════════╪═══════════════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ A                 │
        │ 1          ┆ 2021-01-05 00:00:00 ┆ Delta +365 days   │
        │ 1          ┆ 2021-01-05 00:00:00 ┆ Delta +6 days     │
        │ 1          ┆ 2021-01-05 00:00:00 ┆ B                 │
        │ 1          ┆ 2021-01-06 00:00:00 ┆ Delta +1 days     │
        │ 1          ┆ 2021-01-06 00:00:00 ┆ C                 │
        └────────────┴─────────────────────┴───────────────────┘
    """

    header = stage_cfg.get("header", "")
    delimiter = stage_cfg.get("delimiter", "//")
    header_with_delimiter = f"{header}{delimiter}" if header else header
    max_delta_days = int(stage_cfg.get("max_delta_days", 365))

    if max_delta_days <= 0:
        msg = "max_delta_days must be a positive integer"
        raise ValueError(msg)

    def add_delta_rows_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Insert Delta rows on subject or date changes, skipping the initial +0 day row.
        Assumes df is sorted by subject_id and time ascending
        """
        df = ensure_datetime_column(df, "time")

        # ---------------------------------------------------------
        # 1. Store the original row order to interleave Delta rows correctly later.
        # ---------------------------------------------------------
        original_schema = df.collect_schema()
        df = df.with_row_index("__row_idx")

        # ---------------------------------------------------------
        # 2. DELTA CALCULATION
        # ---------------------------------------------------------
        # Calculate the number of days between the current row and the previous row
        # for the SAME subject.
        # The first row of any subject will result in a 'null' delta.
        event_date = pl.col("time").dt.date()
        delta_expr = (event_date - event_date.shift(1).over("subject_id")).dt.total_days()

        # ---------------------------------------------------------
        # 3. GENERATE DELTA ROWS (PARENTS)
        # ---------------------------------------------------------
        parents_df = (
            df.select("subject_id", "time", "__row_idx")
            .with_columns(__delta_days=delta_expr)
            # Filter logic:
            # > 0 ignores 'null' (first row of subject) and '0' (same-day events).
            .filter(pl.col("__delta_days") > 0)
            # Calculate how many "chunks" of max_delta_days fit into the total delta.
            # Example: if delta is 400 and max is 365, we need 2 rows (365 + 36).
            .with_columns(
                __n_chunks=((pl.col("__delta_days") / max_delta_days).ceil().cast(pl.Int64).clip(lower_bound=1))
            )
            # Create a list of indices [0, 1, ... n] and explode them into separate rows.
            .with_columns(__chunk_idx=pl.int_ranges(0, pl.col("__n_chunks")))
            .explode("__chunk_idx")
            # Assign the day count to each chunk.
            # Full chunks get 'max_delta_days'; the final chunk gets the remainder.
            .with_columns(
                __delta_chunk=pl.when(pl.col("__chunk_idx") < (pl.col("__n_chunks") - 1))
                .then(pl.lit(max_delta_days, dtype=pl.Int64))
                # For the final chunk, simply subtract the already allocated days
                .otherwise(pl.col("__delta_days") - (pl.col("__chunk_idx") * max_delta_days))
            )
        )

        # ---------------------------------------------------------
        # 4. COLUMN MAPPING & TYPE CASTING
        # ---------------------------------------------------------
        # We rebuild the full schema for our Delta rows.
        # 'code' gets the string, others get typed nulls.
        parent_row_exprs = []
        for col_name, col_dtype in original_schema.items():
            if col_name == "code":
                parent_row_exprs.append(
                    pl.format("{}+{} days", pl.lit(header_with_delimiter), pl.col("__delta_chunk")).alias("code")
                )
            elif col_name in ["subject_id", "time"]:
                parent_row_exprs.append(pl.col(col_name))
            else:
                # Important: lit(None) must be cast to the original column's type.
                parent_row_exprs.append(pl.lit(None, dtype=col_dtype).alias(col_name))

        parents_final = parents_df.select(
            "__row_idx",
            *parent_row_exprs,
            "__chunk_idx",
            __is_actual=pl.lit(False),  # Flag to ensure Delta rows sort BEFORE actual rows
        )

        # ---------------------------------------------------------
        # 5. MERGE AND SORT
        # ---------------------------------------------------------
        actual_final = df.with_columns(__chunk_idx=pl.lit(0, dtype=pl.Int64), __is_actual=pl.lit(True))

        return (
            # Combine the new Delta rows with the original data.
            pl.concat([parents_final, actual_final], how="vertical")
            # Sort by:
            # 1. Original row position (__row_idx)
            # 2. Delta vs Actual (__is_actual: False sorts before True)
            # 3. Chunk order (__chunk_idx: 0, 1, 2...)
            .sort(["__row_idx", "__is_actual", "__chunk_idx"])
            # Clean up temporary helper columns.
            .drop("__row_idx", "__chunk_idx", "__is_actual")
        )

    return add_delta_rows_fn
