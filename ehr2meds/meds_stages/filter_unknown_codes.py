"""Remove events whose code contains a configured unknown token."""

import polars as pl
from collections.abc import Callable
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def filter_unknown_codes(data: pl.LazyFrame, unknown_tokens: list[str]) -> pl.LazyFrame:
    """Remove codes containing an unknown ``//``-separated component."""
    if not unknown_tokens:
        return data

    has_unknown_token = pl.col(DataSchema.code_name).str.split("//").list.eval(pl.element().is_in(unknown_tokens)).list.any()
    return data.filter(~has_unknown_token)


def filter_unknown_codes_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the unknown-code filter from the configured tokens."""
    unknown_tokens = list(stage_cfg.unknown_tokens)

    def transform(data: pl.LazyFrame) -> pl.LazyFrame:
        return filter_unknown_codes(data, unknown_tokens)

    return transform


stage = Stage.register(
    is_metadata=False,
    default_config=Path("configs/MEDS/default_filter_unknown_codes.yaml"),
)(filter_unknown_codes_fntr)
