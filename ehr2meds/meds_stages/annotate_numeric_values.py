"""Apply frozen combined-binning metadata without changing MEDS codes or values."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from ehr2meds.meds_stages.aggregate_numeric_metadata import (
    BIN_EDGES,
    BIN_REPRESENTATIVES,
    HARD_MAXIMUM,
    HARD_MINIMUM,
    LOWER_BOUND,
    UPPER_BOUND,
)
from meds import CodeMetadataSchema, DataSchema
from MEDS_transforms.stages import Stage
from MEDS_transforms.utils import PKG_PFX, resolve_pkg_path
from omegaconf import DictConfig
from pathlib import Path

CODE = DataSchema.code_name
VALUE = DataSchema.numeric_value_name
NORMALIZED_VALUE = "numeric_value_normalized"
BIN_INDEX = "numeric_value_bin"
BINNED_VALUE = "numeric_value_binned"
DERIVED_COLUMNS = {
    NORMALIZED_VALUE: pl.Float32,
    BIN_INDEX: pl.Int32,
    BINNED_VALUE: pl.Float32,
}
TRANSFORM_COLUMNS = [LOWER_BOUND, UPPER_BOUND, BIN_EDGES, BIN_REPRESENTATIVES]
BOUND_COLUMNS = [HARD_MINIMUM, HARD_MAXIMUM]
METADATA_COLUMNS = [*TRANSFORM_COLUMNS, *BOUND_COLUMNS]


def _find_bin(row: dict[str, object]) -> int | None:
    """Return the right-sided insertion index of a value in its bin edges."""
    edges = row[BIN_EDGES]
    normalized = row["normalized"]
    if edges is None or normalized is None:
        return None
    return sum(edge <= normalized for edge in edges)


def _load_external_metadata(filepath: str) -> pl.DataFrame:
    """Load frozen numeric metadata from JSON, Parquet, or a MEDS metadata directory."""
    path = resolve_pkg_path(filepath) if filepath.startswith(PKG_PFX) else Path(filepath)
    if path.is_dir():
        path = path / "codes.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"numeric_metadata_filepath '{filepath}' does not exist")
    match path.suffix.lower():
        case ".parquet":
            return pl.read_parquet(path)
        case ".json":
            return pl.read_json(path)
        case _:
            raise ValueError(
                "numeric_metadata_filepath must be a JSON or Parquet file, or a directory containing codes.parquet"
            )


def _with_optional_bounds(metadata: pl.DataFrame) -> pl.DataFrame:
    """Make absent hard bounds into nullable columns for uniform annotation."""
    missing_bounds = [column for column in BOUND_COLUMNS if column not in metadata.columns]
    return metadata.with_columns(*(pl.lit(None, dtype=pl.Float64).alias(column) for column in missing_bounds))


def _validate_metadata(metadata: pl.DataFrame, *, key: list[str], label: str = "numeric") -> None:
    """Require the key and transform columns needed to annotate events."""
    required = [*key, *TRANSFORM_COLUMNS]
    missing = set(required) - set(metadata.columns)
    if missing:
        raise ValueError(f"{label} metadata is missing columns: {sorted(missing)}")


def _metadata_for_annotation(metadata: pl.DataFrame, *, key: list[str]) -> pl.LazyFrame:
    """Validate metadata, add absent optional bounds, and select join columns."""
    _validate_metadata(metadata, key=key)
    return _with_optional_bounds(metadata).lazy().select(*key, *METADATA_COLUMNS)


def _combine_numeric_metadata(
    fitted_metadata: pl.DataFrame,
    external_metadata: pl.DataFrame,
    *,
    key: list[str],
) -> pl.DataFrame:
    """Overlay external transforms on local transforms, matching by code key."""
    for label, metadata in (("fitted", fitted_metadata), ("external", external_metadata)):
        _validate_metadata(metadata, key=key, label=label)

    # External rows come first and therefore win. Local rows provide fallback.
    # Useful when for instance concepts are not present in the external metadata.
    external_metadata = _with_optional_bounds(external_metadata)
    fitted_metadata = _with_optional_bounds(fitted_metadata)
    columns = [*key, *METADATA_COLUMNS]

    return (
        pl.concat(
            [external_metadata.select(columns), fitted_metadata.select(columns)],
            how="vertical_relaxed",
        )
        .unique(subset=key, keep="first", maintain_order=True)
        .sort(key)
    )


def _usable_value() -> pl.Expr:
    """Identify finite values that satisfy optional hard bounds."""
    is_finite = pl.col(VALUE).is_not_null() & pl.col(VALUE).is_finite()
    above_minimum = pl.col(HARD_MINIMUM).is_null() | (pl.col(VALUE) >= pl.col(HARD_MINIMUM))
    below_maximum = pl.col(HARD_MAXIMUM).is_null() | (pl.col(VALUE) <= pl.col(HARD_MAXIMUM))
    has_transform = pl.col(LOWER_BOUND).is_not_null()
    return is_finite & above_minimum & below_maximum & has_transform


def _normalized_value() -> pl.Expr:
    """Clip to fitted percentiles and normalize to the interval [0, 1]."""
    clipped = pl.col(VALUE).clip(pl.col(LOWER_BOUND), pl.col(UPPER_BOUND))
    return (
        pl.when(pl.col(UPPER_BOUND) <= pl.col(LOWER_BOUND))
        .then(0.0)
        .otherwise((clipped - pl.col(LOWER_BOUND)) / (pl.col(UPPER_BOUND) - pl.col(LOWER_BOUND)))
    )


def _bin_index(normalized: pl.Expr) -> pl.Expr:
    """Find each normalized value's right-sided quantile bin."""
    # Polars cannot reference the row's normalized scalar inside list.eval.
    # The edge lists are small and bounded, so a row-local struct is clear and safe.
    return pl.struct(pl.col(BIN_EDGES), normalized.alias("normalized")).map_elements(
        _find_bin,
        return_dtype=pl.Int32,
    )


def _annotation_columns() -> list[pl.Expr]:
    """Build the three numeric columns added to each event."""
    usable = _usable_value()
    normalized = _normalized_value()
    bin_index = _bin_index(normalized)

    return [
        pl.when(usable).then(normalized).cast(pl.Float32).alias(NORMALIZED_VALUE),
        pl.when(usable).then(bin_index).cast(pl.Int32).alias(BIN_INDEX),
        pl.when(usable)
        .then(pl.col(BIN_REPRESENTATIVES).list.get(bin_index, null_on_oob=True))
        .cast(pl.Float32)
        .alias(BINNED_VALUE),
    ]


def annotate_numeric_values(data: pl.LazyFrame, metadata: pl.DataFrame, *, key: list[str]) -> pl.LazyFrame:
    """Apply frozen transforms while preserving the original event columns.

    Numeric rows for unseen concepts and nonnumeric rows receive null derived
    values. The base ``code`` and ``numeric_value`` are never changed.
    """
    transforms = _metadata_for_annotation(metadata, key=key)
    return data.join(transforms, on=key, how="left").with_columns(_annotation_columns()).drop(*METADATA_COLUMNS)


@Stage.register(output_schema_updates=DERIVED_COLUMNS)
def annotate_numeric_values_fntr(
    stage_cfg: DictConfig,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the shard annotator from local and optional external metadata.

    ``numeric_metadata_filepath`` may point to a fitted numeric-metadata JSON,
    another dataset's ``metadata/codes.parquet``, or its ``metadata`` directory.
    External transforms override locally fitted transforms for matching keys.
    """
    key = [CodeMetadataSchema.code_name, *(code_modifiers or [])]
    metadata = code_metadata
    external_filepath = stage_cfg.get("numeric_metadata_filepath")
    if external_filepath:
        metadata = _combine_numeric_metadata(
            code_metadata,
            _load_external_metadata(str(external_filepath)),
            key=key,
        )

    def annotate(df: pl.LazyFrame) -> pl.LazyFrame:
        return annotate_numeric_values(df, metadata, key=key)

    return annotate


stage = annotate_numeric_values_fntr
