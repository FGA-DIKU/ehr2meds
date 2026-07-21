"""Fit per-code numeric normalization and binning on training data."""

from __future__ import annotations

import math
import polars as pl
from collections.abc import Callable
from dataclasses import dataclass
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path

CODE = DataSchema.code_name
VALUE = DataSchema.numeric_value_name
TRAINING_VALUES = "numeric_values/training_values"
LOWER_BOUND = "numeric/p1"
UPPER_BOUND = "numeric/p99"
BIN_EDGES = "numeric/bin_edges"
BIN_REPRESENTATIVES = "numeric/bin_representatives"
HARD_MINIMUM = "numeric/hard_minimum"
HARD_MAXIMUM = "numeric/hard_maximum"


@dataclass(frozen=True)
class NumericBinningConfig:
    """Parameters shared by every fitted numeric concept."""

    min_bins: int = 2
    max_bins: int = 100
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    def __post_init__(self) -> None:
        if self.min_bins < 1 or self.max_bins < self.min_bins:
            raise ValueError("bin bounds must satisfy 1 <= min_bins <= max_bins")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")

    @classmethod
    def from_stage_config(cls, stage_cfg: DictConfig) -> "NumericBinningConfig":
        return cls(
            min_bins=int(stage_cfg.get("min_bins", 2)),
            max_bins=int(stage_cfg.get("max_bins", 100)),
            lower_quantile=float(stage_cfg.get("lower_quantile", 0.01)),
            upper_quantile=float(stage_cfg.get("upper_quantile", 0.99)),
        )


def calculate_bin_count(n_unique: int, minimum: int, maximum: int) -> int:
    """Return the bounded adaptive bin count B(N)=1.14*N_unique**0.237."""
    if minimum < 1 or maximum < minimum:
        raise ValueError("bin bounds must satisfy 1 <= min_bins <= max_bins")
    return max(minimum, min(maximum, int(round(1.14 * n_unique**0.237))))


def _read_value_bounds(stage_cfg: DictConfig) -> pl.DataFrame | None:
    """Validate optional per-code hard bounds and return them as a table."""
    configured_bounds = stage_cfg.get("numeric_value_bounds")
    if not configured_bounds:
        return None

    rows = []
    for code, bounds in configured_bounds.items():
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        minimum = float(minimum) if minimum is not None else None
        maximum = float(maximum) if maximum is not None else None

        if minimum is None and maximum is None:
            raise ValueError(f"numeric_value_bounds[{code!r}] must define min, max, or both")
        if minimum is not None and not math.isfinite(minimum):
            raise ValueError(f"numeric_value_bounds[{code!r}].min must be finite")
        if maximum is not None and not math.isfinite(maximum):
            raise ValueError(f"numeric_value_bounds[{code!r}].max must be finite")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"numeric_value_bounds[{code!r}] has min greater than max")

        rows.append({CODE: str(code), HARD_MINIMUM: minimum, HARD_MAXIMUM: maximum})

    return pl.DataFrame(
        rows,
        schema={CODE: pl.String, HARD_MINIMUM: pl.Float64, HARD_MAXIMUM: pl.Float64},
    )


def _keep_values_within_bounds(df: pl.LazyFrame, bounds: pl.DataFrame | None) -> pl.LazyFrame:
    """Keep finite values and, where configured, values within hard bounds."""
    df = df.filter(pl.col(VALUE).is_not_null() & pl.col(VALUE).is_finite())
    if bounds is None:
        return df

    return (
        df.join(bounds.lazy(), on=CODE, how="left")
        .filter(
            (pl.col(HARD_MINIMUM).is_null() | (pl.col(VALUE) >= pl.col(HARD_MINIMUM)))
            & (pl.col(HARD_MAXIMUM).is_null() | (pl.col(VALUE) <= pl.col(HARD_MAXIMUM)))
        )
        .drop(HARD_MINIMUM, HARD_MAXIMUM)
    )


def _fit_transform(values: list[float], config: NumericBinningConfig) -> dict[str, object]:
    """Transform and annotate one code."""
    series = pl.Series(sorted(values), dtype=pl.Float64)
    lower = float(series.quantile(config.lower_quantile, interpolation="linear"))
    upper = float(series.quantile(config.upper_quantile, interpolation="linear"))
    n_bins = calculate_bin_count(series.n_unique(), config.min_bins, config.max_bins)

    if upper <= lower:
        edges: list[float] = []
        representatives = [0.0]
    else:
        clipped = series.clip(lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        quantile_edges = [float(normalized.quantile(i / n_bins, interpolation="linear")) for i in range(1, n_bins)]
        edges = sorted(set(quantile_edges))
        boundaries = [0.0, *edges, 1.0]
        representatives = [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])]

    return {
        LOWER_BOUND: lower,
        UPPER_BOUND: upper,
        BIN_EDGES: edges,
        BIN_REPRESENTATIVES: representatives,
    }


def mapper_fntr(stage_cfg: DictConfig, code_modifiers: list[str] | None = None) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Collect valid training values (``train_only`` selects shards)."""
    key = [CODE, *(code_modifiers or [])]
    bounds = _read_value_bounds(stage_cfg)

    def mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            _keep_values_within_bounds(df, bounds)
            .group_by(key)
            .agg(pl.col(VALUE).cast(pl.Float64).alias(TRAINING_VALUES))
            .sort(key)
        )

    return mapper


def fit_numeric_metadata(
    *dfs: pl.DataFrame | pl.LazyFrame,
    key: list[str],
    min_bins: int = 2,
    max_bins: int = 100,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pl.DataFrame:
    """Combine training shards and make per-code transformations.

    The intended behavior is to receive only training shards, configured
    by running the command with ``train_only: true``.
    """
    config = NumericBinningConfig(
        min_bins=min_bins,
        max_bins=max_bins,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    frames = [df.collect() if isinstance(df, pl.LazyFrame) else df for df in dfs]
    if not frames:
        return pl.DataFrame()
    values_by_key: dict[tuple, list[float]] = {}
    for frame in frames:
        for row in frame.iter_rows(named=True):
            group = tuple(row[c] for c in key)
            finite_values = (float(value) for value in row[TRAINING_VALUES] if value is not None)
            values_by_key.setdefault(group, []).extend(value for value in finite_values if math.isfinite(value))

    records = []
    for group in sorted(values_by_key, key=lambda x: tuple("" if v is None else str(v) for v in x)):
        values = sorted(values_by_key[group])
        record = dict(zip(key, group, strict=True))
        record.update(_fit_transform(values, config))
        records.append(record)
    return pl.DataFrame(records).sort(key)


def reducer_fntr(stage_cfg: DictConfig, code_modifiers: list[str] | None = None) -> Callable[..., pl.DataFrame]:
    """Build the global reducer usin one configuration."""
    key = [CODE, *(code_modifiers or [])]
    config = NumericBinningConfig.from_stage_config(stage_cfg)
    bounds = _read_value_bounds(stage_cfg)
    configured_output = stage_cfg.get("numeric_metadata_output_filepath")
    output_filepath = (
        Path(str(configured_output))
        if configured_output
        else Path(str(stage_cfg.reducer_output_dir)) / "numeric_metadata.json"
    )

    def reducer(*dfs: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        metadata = fit_numeric_metadata(
            *dfs,
            key=key,
            min_bins=config.min_bins,
            max_bins=config.max_bins,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
        )
        if bounds is not None and not metadata.is_empty():
            metadata = metadata.join(bounds, on=CODE, how="left")
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_json(output_filepath)
        return metadata

    return reducer


stage = Stage.register(map_fn=mapper_fntr, reduce_fn=reducer_fntr)
