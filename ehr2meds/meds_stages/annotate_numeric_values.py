"""Apply frozen combined-binning metadata without changing MEDS codes or values."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from ehr2meds.meds_stages.aggregate_numeric_metadata import (
    BIN_EDGES,
    BIN_REPRESENTATIVES,
    IS_CONSTANT,
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
DERIVED_COLUMNS = {
    "numeric_value_normalized": pl.Float32,
    "numeric_value_bin": pl.Int32,
    "numeric_value_binned": pl.Float32,
    "numeric_value_present": pl.Boolean,
    "numeric_value_was_clipped": pl.Boolean,
}
REQUIRED_METADATA_COLUMNS = [LOWER_BOUND, UPPER_BOUND, BIN_EDGES, BIN_REPRESENTATIVES, IS_CONSTANT]


def _find_bin(row: dict[str, object]) -> int | None:
    """Return the right-sided insertion index of a value in its bin edges."""
    edges = row[BIN_EDGES]
    normalized = row["normalized"]
    if edges is None or normalized is None:
        return None
    return sum(edge <= normalized for edge in edges)


def _load_external_metadata(filepath: str) -> pl.DataFrame:
    """Load frozen numeric metadata from a Parquet file or MEDS metadata directory."""
    path = resolve_pkg_path(filepath) if filepath.startswith(PKG_PFX) else Path(filepath)
    if path.is_dir():
        path = path / "codes.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"numeric_metadata_filepath '{filepath}' does not exist")
    if path.suffix.lower() != ".parquet":
        raise ValueError("numeric_metadata_filepath must be a Parquet file or a directory containing codes.parquet")
    return pl.read_parquet(path)


def _combine_numeric_metadata(
    fitted_metadata: pl.DataFrame,
    external_metadata: pl.DataFrame,
    *,
    key: list[str],
) -> pl.DataFrame:
    """Overlay external transforms on local transforms, matching by code key."""
    required = [*key, *REQUIRED_METADATA_COLUMNS]
    for label, metadata in (("fitted", fitted_metadata), ("external", external_metadata)):
        missing = set(required) - set(metadata.columns)
        if missing:
            raise ValueError(f"{label} numeric metadata is missing columns: {sorted(missing)}")

    # External rows come first and therefore win. Local rows provide a useful
    # fallback for concepts not present in the external dataset.
    return (
        pl.concat(
            [external_metadata.select(required), fitted_metadata.select(required)],
            how="vertical_relaxed",
        )
        .unique(subset=key, keep="first", maintain_order=True)
        .sort(key)
    )


def annotate_numeric_values(data: pl.LazyFrame, metadata: pl.DataFrame, *, key: list[str]) -> pl.LazyFrame:
    """Apply frozen transforms while preserving the original event columns.

    Numeric rows for unseen concepts retain ``numeric_value_present=True`` but
    receive null derived values because no training-only transform exists.
    Nonnumeric rows receive ``numeric_value_present=False`` and null derived
    values. The base ``code`` and canonical ``numeric_value`` are never changed.
    """
    available = set(metadata.columns)
    missing = set(REQUIRED_METADATA_COLUMNS) - available
    if missing:
        raise ValueError(f"numeric metadata is missing columns: {sorted(missing)}")

    frozen_metadata = metadata.lazy().select(*key, *REQUIRED_METADATA_COLUMNS)
    joined = data.join(frozen_metadata, on=key, how="left")

    value_is_present = pl.col(VALUE).is_not_null() & pl.col(VALUE).is_finite()
    transform_is_fitted = pl.col(LOWER_BOUND).is_not_null()
    value_can_be_annotated = value_is_present & transform_is_fitted
    clipped = pl.col(VALUE).clip(pl.col(LOWER_BOUND), pl.col(UPPER_BOUND))
    normalized = (
        pl.when(pl.col(IS_CONSTANT))
        .then(0.0)
        .otherwise((clipped - pl.col(LOWER_BOUND)) / (pl.col(UPPER_BOUND) - pl.col(LOWER_BOUND)))
    )
    # Polars does not currently permit referencing the row's normalized scalar
    # from inside ``list.eval``. A struct expression keeps this operation local
    # to the two small per-row values (the adaptive edge list is bounded).
    bin_index = (
        pl.struct(pl.col(BIN_EDGES), normalized.alias("normalized"))
        .map_elements(_find_bin, return_dtype=pl.Int32)
        .cast(pl.Int32)
    )

    return joined.with_columns(
        pl.when(value_can_be_annotated).then(normalized).cast(pl.Float32).alias("numeric_value_normalized"),
        pl.when(value_can_be_annotated).then(bin_index).cast(pl.Int32).alias("numeric_value_bin"),
        pl.when(value_can_be_annotated)
        .then(pl.col(BIN_REPRESENTATIVES).list.get(bin_index, null_on_oob=True))
        .cast(pl.Float32)
        .alias("numeric_value_binned"),
        value_is_present.cast(pl.Boolean).alias("numeric_value_present"),
        pl.when(value_can_be_annotated).then(pl.col(VALUE) != clipped).cast(pl.Boolean).alias("numeric_value_was_clipped"),
    ).drop(REQUIRED_METADATA_COLUMNS)


@Stage.register(output_schema_updates=DERIVED_COLUMNS)
def annotate_numeric_values_fntr(
    stage_cfg: DictConfig,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the shard annotator from local and optional external metadata.

    ``numeric_metadata_filepath`` may point directly to another dataset's
    ``metadata/codes.parquet`` or to its ``metadata`` directory. External
    transforms override locally fitted transforms for matching code keys.
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
