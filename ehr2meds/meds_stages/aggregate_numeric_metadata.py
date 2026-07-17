"""Fit leakage-safe, per-code numeric transforms on training shards."""

from __future__ import annotations

import math
import polars as pl
from collections.abc import Callable
from dataclasses import dataclass
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

CODE = DataSchema.code_name
VALUE = DataSchema.numeric_value_name
VALUES = "numeric_values/training_values"
VERSION = "numeric-combined-binning-v1"
TRAINING_COUNT = "numeric/training_count"
LOWER_BOUND = "numeric/p1"
UPPER_BOUND = "numeric/p99"
BIN_EDGES = "numeric/bin_edges"
BIN_REPRESENTATIVES = "numeric/bin_representatives"
REQUESTED_BIN_COUNT = "numeric/requested_bin_count"
EFFECTIVE_BIN_COUNT = "numeric/effective_bin_count"
IS_CONSTANT = "numeric/is_constant"
CONFIGURATION_VERSION = "numeric/configuration_version"


@dataclass(frozen=True)
class NumericBinningConfig:
    """Validated configuration used to fit every laboratory concept."""

    min_bins: int = 2
    max_bins: int = 100
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    version: str = VERSION

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
            version=str(stage_cfg.get("configuration_version", VERSION)),
        )


def requested_bin_count(n: int, minimum: int, maximum: int) -> int:
    """Return the bounded adaptive bin count B(N)=1.14*N**0.237."""
    if minimum < 1 or maximum < minimum:
        raise ValueError("bin bounds must satisfy 1 <= min_bins <= max_bins")
    return max(minimum, min(maximum, int(round(1.14 * n**0.237))))


def _bin_definition(values: list[float], config: NumericBinningConfig) -> dict[str, object]:
    """Fit one concept and return its frozen normalization/binning fields."""
    series = pl.Series(sorted(values), dtype=pl.Float64)
    lower = float(series.quantile(config.lower_quantile, interpolation="linear"))
    upper = float(series.quantile(config.upper_quantile, interpolation="linear"))
    requested = requested_bin_count(len(values), config.min_bins, config.max_bins)
    is_constant = upper <= lower

    if is_constant:
        edges: list[float] = []
        representatives = [0.0]
    else:
        clipped = series.clip(lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        quantile_edges = [float(normalized.quantile(i / requested, interpolation="linear")) for i in range(1, requested)]
        edges = sorted(set(quantile_edges))
        boundaries = [0.0, *edges, 1.0]
        representatives = [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])]

    return {
        TRAINING_COUNT: len(values),
        LOWER_BOUND: lower,
        UPPER_BOUND: upper,
        BIN_EDGES: edges,
        BIN_REPRESENTATIVES: representatives,
        REQUESTED_BIN_COUNT: requested,
        EFFECTIVE_BIN_COUNT: len(representatives),
        IS_CONSTANT: is_constant,
    }


def mapper_fntr(stage_cfg: DictConfig, code_modifiers: list[str] | None = None) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Collect raw finite values. The framework's ``train_only`` selects shards."""
    key = [CODE, *(code_modifiers or [])]

    def mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.filter(pl.col(VALUE).is_not_null() & pl.col(VALUE).is_finite())
            .group_by(key)
            .agg(pl.col(VALUE).cast(pl.Float64).alias(VALUES))
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
    configuration_version: str = VERSION,
) -> pl.DataFrame:
    """Reduce arbitrary shard layouts into deterministic frozen transforms.

    The caller controls which shards are supplied. In the pipeline this function
    receives only training shards because the stage is configured with
    ``train_only: true``.
    """
    config = NumericBinningConfig(
        min_bins=min_bins,
        max_bins=max_bins,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        version=configuration_version,
    )
    frames = [df.collect() if isinstance(df, pl.LazyFrame) else df for df in dfs]
    if not frames:
        return pl.DataFrame()
    values_by_key: dict[tuple, list[float]] = {}
    for frame in frames:
        for row in frame.iter_rows(named=True):
            group = tuple(row[c] for c in key)
            values_by_key.setdefault(group, []).extend(float(v) for v in row[VALUES] if v is not None and math.isfinite(v))

    records = []
    for group in sorted(values_by_key, key=lambda x: tuple("" if v is None else str(v) for v in x)):
        values = sorted(values_by_key[group])
        record = dict(zip(key, group, strict=True))
        record.update(_bin_definition(values, config))
        record[CONFIGURATION_VERSION] = config.version
        records.append(record)
    return pl.DataFrame(records).sort(key)


def reducer_fntr(stage_cfg: DictConfig, code_modifiers: list[str] | None = None) -> Callable[..., pl.DataFrame]:
    """Build the global reducer using one immutable, validated configuration."""
    key = [CODE, *(code_modifiers or [])]
    config = NumericBinningConfig.from_stage_config(stage_cfg)

    def reducer(*dfs: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        return fit_numeric_metadata(
            *dfs,
            key=key,
            min_bins=config.min_bins,
            max_bins=config.max_bins,
            lower_quantile=config.lower_quantile,
            upper_quantile=config.upper_quantile,
            configuration_version=config.version,
        )

    return reducer


stage = Stage.register(map_fn=mapper_fntr, reduce_fn=reducer_fntr)
