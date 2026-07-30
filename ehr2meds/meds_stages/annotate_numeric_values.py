"""Apply frozen combined-binning metadata without changing MEDS codes or values."""

from __future__ import annotations

import polars as pl
from collections.abc import Callable
from ehr2meds.io_utils import load_frame, resolve_resource_path
from meds import CodeMetadataSchema, DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def find_bin(row: dict[str, object]) -> int | None:
    """Return the right-sided insertion index of a value in its bin edges."""
    edges = row["edges"]
    normalized = row["normalized"]
    if edges is None or normalized is None:
        return None
    return sum(edge <= normalized for edge in edges)


def load_external_metadata(filepath: str) -> pl.DataFrame:
    """Load frozen numeric metadata from JSON, Parquet, or a MEDS metadata directory."""
    path = resolve_resource_path(filepath)
    if path.is_dir():
        path = path / "codes.parquet"
    return load_frame(str(path), "numeric_metadata_filepath")


def prepare_metadata(
    metadata: pl.DataFrame,
    key: list[str],
    transform_columns: list[str],
    bound_columns: list[str],
    label: str = "numeric",
) -> pl.DataFrame:
    """Validate and select the metadata required for annotation."""
    missing = set(key + transform_columns) - set(metadata.columns)
    if missing:
        raise ValueError(f"{label} metadata is missing columns: {sorted(missing)}")

    missing_bounds = [column for column in bound_columns if column not in metadata.columns]
    metadata = metadata.with_columns([pl.lit(None, dtype=pl.Float64).alias(column) for column in missing_bounds])
    return metadata.select(key + transform_columns + bound_columns)


def prepare_numeric_metadata(
    fitted_metadata: pl.DataFrame,
    external_filepath: str | None,
    key: list[str],
    transform_columns: list[str],
    bound_columns: list[str],
) -> pl.DataFrame:
    """Use the locally fitted numeric metadata if it exists; otherwise fall back to an external source."""
    if set(key + transform_columns).issubset(fitted_metadata.columns):
        return prepare_metadata(fitted_metadata, key=key, transform_columns=transform_columns, bound_columns=bound_columns)

    if not external_filepath:
        missing = sorted(set(key + transform_columns) - set(fitted_metadata.columns))
        raise ValueError(f"fitted numeric metadata is missing columns: {missing}")
    external_metadata = load_external_metadata(str(external_filepath))
    return prepare_metadata(
        external_metadata, key=key, transform_columns=transform_columns, bound_columns=bound_columns, label="external"
    )


def is_usable(
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


def normalize(value: pl.Expr, lower_bound: pl.Expr, upper_bound: pl.Expr) -> pl.Expr:
    """Clip to fitted percentiles and normalize to the interval [0, 1]."""
    clipped = value.clip(lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    return pl.when(upper_bound <= lower_bound).then(0.0).otherwise(normalized)


def calculate_bin_index(normalized: pl.Expr, edges: pl.Expr) -> pl.Expr:
    """Find each normalized value's right-sided quantile bin."""
    # Polars cannot reference the row's normalized scalar inside list.eval.
    # The edge lists are small and bounded, so a row-local struct is clear and safe.
    row = pl.struct(edges.alias("edges"), normalized.alias("normalized"))
    return row.map_elements(
        find_bin,
        return_dtype=pl.Int32,
    )


def annotate_numeric_values(
    data: pl.LazyFrame,
    metadata: pl.DataFrame,
    key: list[str],
    transform_columns: list[str],
    bound_columns: list[str],
    derived_roles: list[str],
    lower_bound_column: str,
    upper_bound_column: str,
    bin_edges_column: str,
    bin_representatives_column: str,
    hard_minimum_column: str,
    hard_maximum_column: str,
    normalized_column: str,
    bin_index_column: str,
    binned_column: str,
) -> pl.LazyFrame:
    """Apply frozen transforms while preserving the original event columns.

    Numeric rows for unseen concepts and nonnumeric rows receive null derived
    values. The base ``code`` and ``numeric_value`` are never changed.
    """
    transforms = prepare_metadata(
        metadata,
        key=key,
        transform_columns=transform_columns,
        bound_columns=bound_columns,
    ).lazy()
    annotated = data.join(transforms, on=key, how="left")

    value = pl.col(DataSchema.numeric_value_name)
    lower_bound = pl.col(lower_bound_column)
    upper_bound = pl.col(upper_bound_column)
    hard_minimum = pl.col(hard_minimum_column) if hard_minimum_column in bound_columns else pl.lit(None)
    hard_maximum = pl.col(hard_maximum_column) if hard_maximum_column in bound_columns else pl.lit(None)

    usable = is_usable(
        value=value,
        hard_minimum=hard_minimum,
        hard_maximum=hard_maximum,
        lower_bound=lower_bound,
    )
    normalized = normalize(value=value, lower_bound=lower_bound, upper_bound=upper_bound)
    bin_index = calculate_bin_index(normalized=normalized, edges=pl.col(bin_edges_column))
    binned = pl.col(bin_representatives_column).list.get(bin_index, null_on_oob=True)

    derived_values = {
        "normalized": normalized.cast(pl.Float32),
        "bin_index": bin_index.cast(pl.Int32),
        "binned": binned.cast(pl.Float32),
    }
    derived_names = {
        "normalized": normalized_column,
        "bin_index": bin_index_column,
        "binned": binned_column,
    }
    derived_columns = [pl.when(usable).then(derived_values[role]).alias(derived_names[role]) for role in derived_roles]

    annotated = annotated.with_columns(derived_columns)
    return annotated.drop(transform_columns + bound_columns)


@Stage.register(
    default_config=Path("configs/MEDS/default_numeric_values.yaml"),
)
def annotate_numeric_values_fntr(
    stage_cfg: DictConfig,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the shard annotator from the locally fitted metadata, or an external source if none was fitted.

    ``numeric_metadata_filepath`` may point to a fitted numeric-metadata JSON,
    another dataset's ``metadata/codes.parquet``, or its ``metadata`` directory.
    It's a fallback, not an override: whenever ``aggregate_numeric_metadata``
    ran locally, that metadata is used and ``numeric_metadata_filepath`` is
    ignored.
    """
    key = [CodeMetadataSchema.code_name] + list(code_modifiers or [])
    columns = stage_cfg.numeric_value_columns
    groups = stage_cfg.numeric_value_column_groups
    transform_columns = [columns[role] for role in groups.transform]
    bound_columns = [columns[role] for role in groups.bounds]
    derived_roles = list(groups.derived)
    metadata = prepare_numeric_metadata(
        code_metadata,
        stage_cfg.get("numeric_metadata_filepath"),
        key=key,
        transform_columns=transform_columns,
        bound_columns=bound_columns,
    )

    def annotate(df: pl.LazyFrame) -> pl.LazyFrame:
        return annotate_numeric_values(
            df,
            metadata,
            key=key,
            transform_columns=transform_columns,
            bound_columns=bound_columns,
            derived_roles=derived_roles,
            lower_bound_column=columns.lower_bound,
            upper_bound_column=columns.upper_bound,
            bin_edges_column=columns.bin_edges,
            bin_representatives_column=columns.bin_representatives,
            hard_minimum_column=columns.hard_minimum,
            hard_maximum_column=columns.hard_maximum,
            normalized_column=columns.normalized,
            bin_index_column=columns.bin_index,
            binned_column=columns.binned,
        )

    return annotate


stage = annotate_numeric_values_fntr
