#!/usr/bin/env python
"""Custom MAP stage that orders rows by prefix priority."""

import re
from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig


@Stage.register(is_metadata=False)
def order_by_code_prefix(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Return a function that orders rows by prefix priority.

    Rows are ordered based on which prefix they start with from the priority list.
    Prefixes earlier in the list have higher priority (lower sort value).
    Rows that don't match any prefix are given the lowest priority.
    The sorting is done on the following columns in order:
    - subject_id
    - time
    - _prefix_priority
    - code
    - numeric_value

    Args:
        stage_cfg: Configuration containing:
            - prefix_priority: List of prefix strings in priority order (first =
              highest priority).

    Returns:
        Function that orders the DataFrame by prefix priority.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1],
        ...     "time": [1, 1, 1, 1],
        ...     "code": [
        ...         "ADMISSION//Emergency",
        ...         "LAB//Glucose",
        ...         "MEDICATION//Aspirin",
        ...         "UNKNOWN//Unknown",
        ...     ],
        ...     "numeric_value": [None, 1.0, None, None]
        ... }).lazy()
        >>> cfg = DictConfig({"prefix_priority": ["ADMISSION", "LAB", "MEDICATION"]})
        >>> fn = order_events(cfg)
        >>> result = fn(df).collect()
        >>> result.select("code")
        shape: (4, 1)
        ┌──────────────────────┐
        │ code                 │
        │ ---                  │
        │ str                  │
        ╞══════════════════════╡
        │ ADMISSION//Emergency │
        │ LAB//Glucose         │
        │ MEDICATION//Aspirin  │
        │ UNKNOWN//Unknown     │
        └──────────────────────┘
    """
    prefix_priority = stage_cfg.get("prefix_priority", [])

    # Create a mapping from prefix to priority (0 = highest priority)
    priority_map = {prefix: i for i, prefix in enumerate(prefix_priority)}
    default_priority = len(prefix_priority)

    def order_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        # Create a priority column based on prefix matching
        # Rows that don't match any prefix get the lowest priority (len(prefix_priority))
        if not prefix_priority:
            priority_expr = pl.lit(default_priority, dtype=pl.Int64)
        else:
            # Build regex for prefix matching. Order matters: leftmost-first match
            # which matches the priority order of the list.
            regex = "^(" + "|".join(re.escape(p) for p in prefix_priority) + ")"
            priority_expr = (
                pl.col("code")
                .str.extract(regex)
                .replace_strict(priority_map, default=default_priority, return_dtype=pl.Int64)
            )

        return (
            df.with_columns(priority_expr.alias("_prefix_priority"))
            .sort(
                [
                    "subject_id",
                    "time",
                    "_prefix_priority",
                    "code",
                    "numeric_value",
                ],
                nulls_last=True,
            )
            .drop("_prefix_priority")
        )

    return order_fn
