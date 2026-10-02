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
    namespace: str,
    separator: str,
) -> pl.LazyFrame:
    """Join normalized non-numeric results onto laboratory codes."""
    value = pl.col(DataSchema.text_value_name).cast(pl.String)
    value = value.str.strip_chars()
    value = value.str.replace_all(r"\s+", " ")
    value = value.str.strip_chars_end(" .;:,!?")
    if value_map:
        value = value.replace(value_map)

    code = pl.col(DataSchema.code_name)
    is_lab = code.str.starts_with(f"{namespace}//")
    is_nonnumeric = pl.col(DataSchema.numeric_value_name).is_null()
    has_text = value.is_not_null() & (value != "")
    eligible = is_lab & is_nonnumeric & has_text
    joined = pl.concat_str(code, pl.lit(separator), value)
    result = pl.when(eligible).then(joined).otherwise(code)
    return data.with_columns(result.alias(DataSchema.code_name))


def join_lab_text_values_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Transform function for joining laboratory text values."""
    value_map = dict(stage_cfg.get("value_map", {}))
    namespace = str(stage_cfg.namespace)
    separator = str(stage_cfg.separator)

    def transform(df: pl.LazyFrame) -> pl.LazyFrame:
        return join_lab_text_values(
            df,
            value_map=value_map,
            namespace=namespace,
            separator=separator,
        )

    return transform


stage = Stage.register(
    is_metadata=False,
    default_config=Path("configs/MEDS/default_lab_text_values.yaml"),
)(join_lab_text_values_fntr)
