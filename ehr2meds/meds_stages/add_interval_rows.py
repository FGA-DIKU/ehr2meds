#!/usr/bin/env python
"""Insert interval header rows across each subject's timeline, including empty gaps."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from ..utils import ensure_datetime_column


@Stage.register(is_metadata=False)
def add_interval_rows(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Insert a header at every interval from each subject's first to last timed event.

    Unlike change-triggered markers, this fills **empty** intervals too: take
    ``truncate(min(time))`` … ``truncate(max(time))`` per subject and emit one
    row per step, even when no real events fall in that period.

    Typical use::

        - add_interval_rows:
            header: "## New Month"
            interval: 1mo

    Args:
        stage_cfg:
            - header: Code for marker rows (default: ``"## Interval"``).
            - interval: Polars range/truncate every-string (``1mo``, ``1y``, ``30d``, …).
              ``time`` is accepted as an alias.
            - delimiter: Joiner between header and optional period label (default: ``"//"``).
            - include_period: If true, append the period start as ``YYYY-MM-DD`` (default: false).

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "time": [datetime(2020, 1, 15), datetime(2020, 3, 3)],
        ...     "code": ["A", "B"],
        ... }).lazy()
        >>> fn = add_interval_rows(DictConfig({
        ...     "header": "## New Month",
        ...     "interval": "1mo",
        ... }))
        >>> fn(df).collect().select("subject_id", "time", "code")
        shape: (5, 3)
        ┌────────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code          │
        │ ---        ┆ ---                 ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str           │
        ╞════════════╪═════════════════════╪═══════════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ ## New Month  │
        │ 1          ┆ 2020-01-15 00:00:00 ┆ A             │
        │ 1          ┆ 2020-02-01 00:00:00 ┆ ## New Month  │
        │ 1          ┆ 2020-03-01 00:00:00 ┆ ## New Month  │
        │ 1          ┆ 2020-03-03 00:00:00 ┆ B             │
        └────────────┴─────────────────────┴───────────────┘
    """

    header = str(stage_cfg.get("header", "## Interval"))
    delimiter = str(stage_cfg.get("delimiter", "//"))
    interval = stage_cfg.get("interval", stage_cfg.get("time", "1mo"))
    if interval is None:
        raise ValueError("add_interval_rows requires `interval` (or `time`), e.g. '1mo' or '30d'")
    interval = str(interval)
    include_period = bool(stage_cfg.get("include_period", False))
    header_with_delimiter = f"{header}{delimiter}" if header else header

    def add_interval_rows_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        df = ensure_datetime_column(df, "time")
        original_schema = df.collect_schema()
        time_dtype = original_schema["time"]

        # One [start, end] span per subject from first/last non-null event times.
        spans = (
            df.filter(pl.col("time").is_not_null())
            .group_by("subject_id")
            .agg(
                start=pl.col("time").min().dt.truncate(interval),
                end=pl.col("time").max().dt.truncate(interval),
            )
            .filter(pl.col("start").is_not_null() & pl.col("end").is_not_null())
            .with_columns(
                period=pl.datetime_ranges(
                    pl.col("start"),
                    pl.col("end"),
                    interval=interval,
                    closed="both",
                )
            )
            .explode("period")
            .select(
                pl.col("subject_id"),
                pl.col("period").cast(time_dtype).alias("time"),
            )
        )

        if include_period:
            code_expr = pl.lit(header_with_delimiter) + pl.col("time").dt.strftime("%Y-%m-%d")
        else:
            code_expr = pl.lit(header)

        marker_exprs: list[pl.Expr] = []
        for col_name, col_dtype in original_schema.items():
            if col_name == "code":
                marker_exprs.append(code_expr.alias("code"))
            elif col_name in {"subject_id", "time"}:
                marker_exprs.append(pl.col(col_name))
            else:
                marker_exprs.append(pl.lit(None, dtype=col_dtype).alias(col_name))

        markers = spans.select(*marker_exprs).with_columns(__is_actual=pl.lit(False))
        actual = df.with_columns(__is_actual=pl.lit(True))

        # Interval markers sort before real events at the same timestamp.
        return (
            pl.concat([markers, actual], how="vertical")
            .sort(["subject_id", "time", "__is_actual"], nulls_last=True)
            .drop("__is_actual")
        )

    return add_interval_rows_fn
