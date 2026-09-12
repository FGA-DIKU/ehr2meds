"""Join categorical laboratory results."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def join_lab_text_values(
    data: pl.LazyFrame,
    value_map: dict[str, str],
    encode_comparisons: bool,
    namespace: str,
    separator: str,
) -> pl.LazyFrame:
    """Join configured qualitative results or comparison directions onto lab codes."""
    value = pl.col(DataSchema.text_value_name).cast(pl.String)
    value = value.str.strip_chars().str.to_uppercase()
    value = value.str.replace_all(r"\s+", " ")
    category = pl.lit(None, dtype=pl.String)
    if value_map:
        is_mapped = value.is_in(list(value_map))
        category = pl.when(is_mapped).then(value.replace(value_map)).otherwise(category)

    if encode_comparisons:
        for operator, label in [("<=", "LE"), (">=", "GE"), ("<", "LT"), (">", "GT")]:
            operand = value.str.slice(len(operator)).str.strip_chars()
            number = operand.str.replace(",", ".", literal=True)
            number = number.cast(pl.Float64, strict=False)
            is_comparison = value.str.starts_with(operator) & number.is_finite()
            category = pl.when(is_comparison).then(pl.lit(label)).otherwise(category)

    code = pl.col(DataSchema.code_name)
    is_lab = code.str.starts_with(f"{namespace}//")
    is_nonnumeric = pl.col(DataSchema.numeric_value_name).is_null()
    eligible = is_lab & is_nonnumeric & category.is_not_null()
    joined = pl.concat_str(code, pl.lit(separator), category)
    result = pl.when(eligible).then(joined).otherwise(code)
    return data.with_columns(result.alias(DataSchema.code_name))


def join_lab_text_values_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Transform function for joining laboratory text values."""
    value_map = dict(stage_cfg.get("value_map", {}))
    encode_comparisons = bool(stage_cfg.get("encode_comparisons", False))
    namespace = str(stage_cfg.namespace)
    separator = str(stage_cfg.separator)

    def transform(df: pl.LazyFrame) -> pl.LazyFrame:
        return join_lab_text_values(
            df,
            value_map=value_map,
            encode_comparisons=encode_comparisons,
            namespace=namespace,
            separator=separator,
        )

    return transform


stage = Stage.register(
    is_metadata=False,
    default_config=Path("configs/MEDS/default_lab_text_values.yaml"),
)(join_lab_text_values_fntr)
