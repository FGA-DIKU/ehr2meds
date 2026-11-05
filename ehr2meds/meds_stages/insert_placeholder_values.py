"""Replace a code placeholder with a numeric bin token or text value."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable, Sequence
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, ListConfig
from pathlib import Path


def normalize_text(value_map: dict[str, str]) -> pl.Expr:
    value = pl.col(DataSchema.text_value_name).cast(pl.String)
    value = value.str.strip_chars()
    value = value.str.replace_all(r"\s+", " ")
    value = value.str.strip_chars_end(" .;:,!?")
    value = value.str.to_lowercase()
    if value_map:
        value = value.replace(value_map)
    return value


def format_numeric_value(numeric: pl.Expr, decimal_places: int | None) -> pl.Expr:
    """Stringify ``numeric_value``, optionally rounded to ``decimal_places``."""
    if decimal_places is None:
        return numeric.cast(pl.String)
    if decimal_places < 0:
        raise ValueError("numeric_decimal_places must be >= 0 or null")
    return numeric.round(decimal_places).cast(pl.String)


def insert_placeholder_values(
    data: pl.LazyFrame,
    *,
    placeholder: str,
    bin_column: str,
    bin_prefix: str,
    value_map: dict[str, str],
    code_prefixes: Sequence[str] = (),
    numeric_decimal_places: int | None = 2,
) -> pl.LazyFrame:
    """Fill ``placeholder`` in ``code`` with a bin token, numeric, or text.

    Preference:
    1. If ``bin_column`` is set, substitute ``{bin_prefix}{bin_index}``.
    2. Else if ``numeric_value`` is set, substitute its string form
       (rounded when ``numeric_decimal_places`` is set).
    3. Else if ``text_value`` is non-empty, substitute the normalized text.
    4. Otherwise leave the code unchanged (placeholder may remain).

    Only rows whose ``code`` contains ``placeholder`` are considered. If
    ``code_prefixes`` is non-empty, the code must also start with one of them.
    """
    code = pl.col(DataSchema.code_name)
    has_placeholder = code.str.contains(placeholder, literal=True)

    if code_prefixes:
        prefix_match = pl.any_horizontal(*(code.str.starts_with(prefix) for prefix in code_prefixes))
    else:
        prefix_match = pl.lit(True)

    text = normalize_text(value_map)
    numeric = pl.col(DataSchema.numeric_value_name)
    has_bin = pl.col(bin_column).is_not_null()
    has_numeric = numeric.is_not_null()
    has_text = text.is_not_null() & (text != "")

    bin_token = pl.concat_str(pl.lit(bin_prefix), pl.col(bin_column).cast(pl.String))
    numeric_token = format_numeric_value(numeric, numeric_decimal_places)
    replacement = (
        pl.when(has_bin)
        .then(bin_token)
        .when(has_numeric)
        .then(numeric_token)
        .when(has_text)
        .then(text)
        .otherwise(pl.lit(None, dtype=pl.String))
    )

    eligible = has_placeholder & prefix_match & replacement.is_not_null()
    updated = code.str.replace(placeholder, replacement, literal=True)
    return data.with_columns(
        pl.when(eligible).then(updated).otherwise(code).alias(DataSchema.code_name)
    )


def insert_placeholder_values_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the placeholder substitution transform."""
    placeholder = str(stage_cfg.get("placeholder", "VALUE_PLACEHOLDER"))
    bin_prefix = str(stage_cfg.get("bin_prefix", "BIN_"))
    value_map = dict(stage_cfg.get("value_map", {}))
    bin_column = str(stage_cfg.numeric_value_columns.bin_index)

    raw_prefixes = stage_cfg.get("code_prefixes", [])
    if isinstance(raw_prefixes, ListConfig | list):
        code_prefixes = [str(prefix) for prefix in raw_prefixes]
    else:
        code_prefixes = []

    raw_decimals = stage_cfg.get("numeric_decimal_places", 2)
    numeric_decimal_places = None if raw_decimals is None else int(raw_decimals)

    def transform(df: pl.LazyFrame) -> pl.LazyFrame:
        return insert_placeholder_values(
            df,
            placeholder=placeholder,
            bin_column=bin_column,
            bin_prefix=bin_prefix,
            value_map=value_map,
            code_prefixes=code_prefixes,
            numeric_decimal_places=numeric_decimal_places,
        )

    return transform


stage = Stage.register(
    is_metadata=False,
    default_config=Path("configs/MEDS/default_insert_placeholder_values.yaml"),
)(insert_placeholder_values_fntr)
