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
EDGES_FIELD = "edges"
NORMALIZED_FIELD = "normalized"


def find_bin(row: dict[str, object]) -> int | None:
    """Return the right-sided insertion index of a value in its bin edges."""
    edges = row[EDGES_FIELD]
    normalized = row[NORMALIZED_FIELD]
    if edges is None or normalized is None:
        return None
    return sum(edge <= normalized for edge in edges)


def load_external_metadata(filepath: str) -> pl.DataFrame:
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


def prepare_metadata(metadata: pl.DataFrame, *, key: list[str], label: str = "numeric") -> pl.DataFrame:
    """Validate and select the metadata required for annotation."""
    missing = set([*key, *TRANSFORM_COLUMNS]) - set(metadata.columns)
    if missing:
        raise ValueError(f"{label} metadata is missing columns: {sorted(missing)}")

    missing_bounds = [column for column in BOUND_COLUMNS if column not in metadata.columns]
    metadata = metadata.with_columns(*(pl.lit(None, dtype=pl.Float64).alias(column) for column in missing_bounds))
    return metadata.select(*key, *METADATA_COLUMNS)


def combine_numeric_metadata(
    fitted_metadata: pl.DataFrame,
    external_metadata: pl.DataFrame,
    *,
    key: list[str],
) -> pl.DataFrame:
    """Overlay external transforms on local transforms, matching by code key."""
    external_metadata = prepare_metadata(external_metadata, key=key, label="external")
    fitted_metadata = prepare_metadata(fitted_metadata, key=key, label="fitted")

    # External rows come first, so they take precedence for matching codes.
    combined = pl.concat([external_metadata, fitted_metadata], how="vertical_relaxed")
    combined = combined.unique(subset=key, keep="first", maintain_order=True)
    return combined.sort(key)


def is_usable(
    *,
    value: pl.Expr,
    hard_minimum: pl.Expr,
    hard_maximum: pl.Expr,
    lower_bound: pl.Expr,
) -> pl.Expr:
    """Identify finite values that satisfy optional hard bounds."""
    is_finite = value.is_not_null() & value.is_finite()
    above_minimum = hard_minimum.is_null() | (value >= hard_minimum)
    below_maximum = hard_maximum.is_null() | (value <= hard_maximum)
    has_transform = lower_bound.is_not_null()
    return is_finite & above_minimum & below_maximum & has_transform


def normalize(*, value: pl.Expr, lower_bound: pl.Expr, upper_bound: pl.Expr) -> pl.Expr:
    """Clip to fitted percentiles and normalize to the interval [0, 1]."""
    clipped = value.clip(lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    return pl.when(upper_bound <= lower_bound).then(0.0).otherwise(normalized)


def calculate_bin_index(*, normalized: pl.Expr, edges: pl.Expr) -> pl.Expr:
    """Find each normalized value's right-sided quantile bin."""
    # Polars cannot reference the row's normalized scalar inside list.eval.
    # The edge lists are small and bounded, so a row-local struct is clear and safe.
    row = pl.struct(edges.alias(EDGES_FIELD), normalized.alias(NORMALIZED_FIELD))
    return row.map_elements(
        find_bin,
        return_dtype=pl.Int32,
    )


def annotation_columns(
    *,
    value: pl.Expr,
    hard_minimum: pl.Expr,
    hard_maximum: pl.Expr,
    lower_bound: pl.Expr,
    upper_bound: pl.Expr,
    bin_edges: pl.Expr,
    bin_representatives: pl.Expr,
) -> list[pl.Expr]:
    """Build the three numeric columns added to each event."""
    usable = is_usable(
        value=value,
        hard_minimum=hard_minimum,
        hard_maximum=hard_maximum,
        lower_bound=lower_bound,
    )
    normalized = normalize(value=value, lower_bound=lower_bound, upper_bound=upper_bound)
    bin_index = calculate_bin_index(normalized=normalized, edges=bin_edges)
    representative = bin_representatives.list.get(bin_index, null_on_oob=True)

    return [
        pl.when(usable).then(normalized).cast(pl.Float32).alias(NORMALIZED_VALUE),
        pl.when(usable).then(bin_index).cast(pl.Int32).alias(BIN_INDEX),
        pl.when(usable).then(representative).cast(pl.Float32).alias(BINNED_VALUE),
    ]


def annotate_numeric_values(data: pl.LazyFrame, metadata: pl.DataFrame, *, key: list[str]) -> pl.LazyFrame:
    """Apply frozen transforms while preserving the original event columns.

    Numeric rows for unseen concepts and nonnumeric rows receive null derived
    values. The base ``code`` and ``numeric_value`` are never changed.
    """
    transforms = prepare_metadata(metadata, key=key).lazy()
    annotated = data.join(transforms, on=key, how="left")
    columns = annotation_columns(
        value=pl.col(DataSchema.numeric_value_name),
        hard_minimum=pl.col(HARD_MINIMUM),
        hard_maximum=pl.col(HARD_MAXIMUM),
        lower_bound=pl.col(LOWER_BOUND),
        upper_bound=pl.col(UPPER_BOUND),
        bin_edges=pl.col(BIN_EDGES),
        bin_representatives=pl.col(BIN_REPRESENTATIVES),
    )
    annotated = annotated.with_columns(columns)
    return annotated.drop(*METADATA_COLUMNS)


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
        external_metadata = load_external_metadata(str(external_filepath))
        metadata = combine_numeric_metadata(code_metadata, external_metadata, key=key)

    def annotate(df: pl.LazyFrame) -> pl.LazyFrame:
        return annotate_numeric_values(df, metadata, key=key)

    return annotate


stage = annotate_numeric_values_fntr
