"""Shared helpers for MEDS structural stages."""

from __future__ import annotations

import polars as pl


def _changed(column: str | pl.Expr) -> pl.Expr:
    """True on the first row and whenever ``column`` differs from the previous row.

    Null equals null (so consecutive null times do not keep firing).
    """
    expr = pl.col(column) if isinstance(column, str) else column
    return ~expr.eq_missing(expr.shift(1))


def ensure_datetime_column(df: pl.LazyFrame, column: str = "time") -> pl.LazyFrame:
    """Ensure ``column`` is datetime[us], parsing strings/dates when needed."""
    dtype = df.collect_schema()[column]
    if isinstance(dtype, pl.Datetime):
        return df
    if dtype == pl.Date:
        return df.with_columns(pl.col(column).cast(pl.Datetime(time_unit="us")))
    return df.with_columns(pl.col(column).str.to_datetime(strict=False, time_unit="us"))


def insert_parent_rows(
    df: pl.LazyFrame,
    *,
    change_cond: pl.Expr,
    code_expr: pl.Expr,
) -> pl.LazyFrame:
    """Insert a parent row with ``code_expr`` before each row where ``change_cond`` is true.

    Parent rows keep ``subject_id`` and ``time``; all other columns are null.
    Assumes ``df`` is already sorted by ``subject_id`` and ``time``.
    """
    original_schema = df.collect_schema()
    indexed = df.with_row_index("__row_idx")

    parent_exprs: list[pl.Expr] = []
    for col_name, col_dtype in original_schema.items():
        if col_name == "code":
            parent_exprs.append(code_expr.alias("code"))
        elif col_name in {"subject_id", "time"}:
            parent_exprs.append(pl.col(col_name))
        else:
            parent_exprs.append(pl.lit(None, dtype=col_dtype).alias(col_name))

    parents = (
        indexed.filter(change_cond)
        .select("__row_idx", *parent_exprs)
        .with_columns(__is_actual=pl.lit(False))
    )
    actual = indexed.with_columns(__is_actual=pl.lit(True))

    return (
        pl.concat([parents, actual], how="vertical")
        .sort(["__row_idx", "__is_actual"])
        .drop("__row_idx", "__is_actual")
    )
