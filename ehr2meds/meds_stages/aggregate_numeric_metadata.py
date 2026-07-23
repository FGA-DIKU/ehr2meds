"""Fit per-code numeric normalization and binning on training data."""

from __future__ import annotations

import math
import polars as pl
from collections.abc import Callable, Mapping
from datetime import datetime
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def calculate_bin_count(n_unique: int, minimum: int, maximum: int) -> int:
    """Return the bounded adaptive bin count B(N)=1.14*N_unique**0.237."""
    estimated = round(1.14 * n_unique**0.237)
    return max(minimum, min(maximum, estimated))


def read_value_bounds(
    configured_bounds: Mapping[str, Mapping[str, float | None]] | None,
    *,
    hard_minimum_column: str,
    hard_maximum_column: str,
) -> pl.DataFrame | None:
    """Validate optional per-code hard bounds and return them as a table."""
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

        rows.append(
            {
                DataSchema.code_name: str(code),
                hard_minimum_column: minimum,
                hard_maximum_column: maximum,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            DataSchema.code_name: pl.String,
            hard_minimum_column: pl.Float64,
            hard_maximum_column: pl.Float64,
        },
    )


def keep_values_within_bounds(
    df: pl.LazyFrame,
    bounds: pl.DataFrame | None,
    *,
    hard_minimum_column: str,
    hard_maximum_column: str,
) -> pl.LazyFrame:
    """Keep finite values and, where configured, values within hard bounds."""
    value = pl.col(DataSchema.numeric_value_name)
    finite_value = value.is_not_null() & value.is_finite()
    df = df.filter(finite_value)
    if bounds is None:
        return df

    hard_minimum = pl.col(hard_minimum_column)
    hard_maximum = pl.col(hard_maximum_column)
    above_minimum = hard_minimum.is_null() | (value >= hard_minimum)
    below_maximum = hard_maximum.is_null() | (value <= hard_maximum)

    df = df.join(bounds.lazy(), on=DataSchema.code_name, how="left")
    df = df.filter(above_minimum & below_maximum)
    return df.drop(hard_minimum_column, hard_maximum_column)


def event_time_filter(configured_cutoff: str | None) -> pl.Expr | None:
    """Restrict fitting to events before an optional calendar date.

    The cutoff is the first excluded date. Null event times cannot be shown to
    precede it, so they are also excluded whenever a cutoff is configured.
    """
    if configured_cutoff is None:
        return None

    try:
        cutoff = datetime.strptime(str(configured_cutoff), "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("event_time_cutoff must be a date in YYYY-MM-DD format") from error

    return pl.col(DataSchema.time_name) < cutoff


def fit_transform(
    values: list[float],
    *,
    lower_bound_column: str,
    upper_bound_column: str,
    bin_edges_column: str,
    bin_representatives_column: str,
    min_bins: int,
    max_bins: int,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, object]:
    """Fit one code's normalization bounds and quantile bins."""
    series = pl.Series(sorted(values), dtype=pl.Float64)
    lower = float(series.quantile(lower_quantile, interpolation="linear"))
    upper = float(series.quantile(upper_quantile, interpolation="linear"))
    n_bins = calculate_bin_count(series.n_unique(), min_bins, max_bins)

    if upper == lower:
        edges: list[float] = []
        representatives = [0.0]
    else:
        normalized = (series.clip(lower, upper) - lower) / (upper - lower)
        quantile_edges = [float(normalized.quantile(index / n_bins, interpolation="linear")) for index in range(1, n_bins)]
        edges = sorted(set(quantile_edges))
        boundaries = [0.0, *edges, 1.0]
        representatives = [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])]

    return {
        lower_bound_column: lower,
        upper_bound_column: upper,
        bin_edges_column: edges,
        bin_representatives_column: representatives,
    }


def mapper_fntr(stage_cfg: DictConfig, code_modifiers: list[str] | None = None) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Collect eligible values (``train_only`` selects the input shards)."""
    key = [DataSchema.code_name, *(code_modifiers or [])]
    columns = stage_cfg.numeric_value_columns
    bounds = read_value_bounds(
        stage_cfg.get("numeric_value_bounds"),
        hard_minimum_column=columns.hard_minimum,
        hard_maximum_column=columns.hard_maximum,
    )
    time_filter = event_time_filter(stage_cfg.get("event_time_cutoff"))

    def mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        if time_filter is not None:
            df = df.filter(time_filter)

        df = keep_values_within_bounds(
            df,
            bounds,
            hard_minimum_column=columns.hard_minimum,
            hard_maximum_column=columns.hard_maximum,
        )
        values = pl.col(DataSchema.numeric_value_name).cast(pl.Float64).alias(columns.training_values)
        return df.group_by(key).agg(values).sort(key)

    return mapper


def fit_numeric_metadata(
    *dfs: pl.DataFrame | pl.LazyFrame,
    key: list[str],
    training_values_column: str,
    lower_bound_column: str,
    upper_bound_column: str,
    bin_edges_column: str,
    bin_representatives_column: str,
    min_bins: int,
    max_bins: int,
    lower_quantile: float,
    upper_quantile: float,
) -> pl.DataFrame:
    """Combine mapped training shards and fit each code's transformation."""
    frames = [df.collect() if isinstance(df, pl.LazyFrame) else df for df in dfs]
    if not frames:
        return pl.DataFrame()

    values_by_key: dict[tuple, list[float]] = {}
    for frame in frames:
        for row in frame.iter_rows(named=True):
            group = tuple(row[column] for column in key)
            finite_values = (float(value) for value in row[training_values_column] if value is not None)
            values_by_key.setdefault(group, []).extend(value for value in finite_values if math.isfinite(value))

    def group_sort_key(group: tuple) -> tuple[str, ...]:
        return tuple("" if value is None else str(value) for value in group)

    records = []
    for group in sorted(values_by_key, key=group_sort_key):
        record = dict(zip(key, group, strict=True))
        transform = fit_transform(
            values_by_key[group],
            lower_bound_column=lower_bound_column,
            upper_bound_column=upper_bound_column,
            bin_edges_column=bin_edges_column,
            bin_representatives_column=bin_representatives_column,
            min_bins=min_bins,
            max_bins=max_bins,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        record.update(transform)
        records.append(record)
    return pl.DataFrame(records).sort(key)


def reducer_fntr(stage_cfg: DictConfig, code_modifiers: list[str] | None = None) -> Callable[..., pl.DataFrame]:
    """Build the global metadata reducer."""
    key = [DataSchema.code_name, *(code_modifiers or [])]
    columns = stage_cfg.numeric_value_columns
    min_bins = int(stage_cfg.min_bins)
    max_bins = int(stage_cfg.max_bins)
    lower_quantile = float(stage_cfg.lower_quantile)
    upper_quantile = float(stage_cfg.upper_quantile)
    if min_bins < 1 or max_bins < min_bins:
        raise ValueError("bin bounds must satisfy 1 <= min_bins <= max_bins")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    bounds = read_value_bounds(
        stage_cfg.get("numeric_value_bounds"),
        hard_minimum_column=columns.hard_minimum,
        hard_maximum_column=columns.hard_maximum,
    )
    configured_output = stage_cfg.get("numeric_metadata_output_filepath")
    if configured_output:
        output_filepath = Path(str(configured_output))
    else:
        output_filepath = Path(str(stage_cfg.reducer_output_dir)) / "numeric_metadata.json"

    def reducer(*dfs: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        metadata = fit_numeric_metadata(
            *dfs,
            key=key,
            training_values_column=columns.training_values,
            lower_bound_column=columns.lower_bound,
            upper_bound_column=columns.upper_bound,
            bin_edges_column=columns.bin_edges,
            bin_representatives_column=columns.bin_representatives,
            min_bins=min_bins,
            max_bins=max_bins,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        if bounds is not None and not metadata.is_empty():
            metadata = metadata.join(bounds, on=DataSchema.code_name, how="left")
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_json(output_filepath)
        return metadata

    return reducer


stage = Stage.register(
    map_fn=mapper_fntr,
    reduce_fn=reducer_fntr,
    default_config=Path("configs/MEDS/default_numeric_values.yaml"),
)
