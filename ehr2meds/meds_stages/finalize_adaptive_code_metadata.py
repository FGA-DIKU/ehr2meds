"""Rewrite MEDS code metadata to the adaptively mapped vocabulary."""

from __future__ import annotations

import polars as pl
from collections.abc import Sequence
from ehr2meds.adaptive_code_mapping import (
    COUNT_COLUMN,
    MAPPED_CODE_COLUMN,
    MAPPED_COUNT_COLUMN,
    MEMBER_COUNT_COLUMN,
    PROFILE_COLUMN,
    REASON_COLUMN,
    prepare_mapping,
)
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def collapse_code_metadata(metadata: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """Rewrite and deterministically collapse code metadata."""
    is_exact_match = "is_exact_match"
    mapped = (
        metadata.join(mapping.select(DataSchema.code_name, MAPPED_CODE_COLUMN), on=DataSchema.code_name, how="left")
        .with_columns(
            pl.coalesce(MAPPED_CODE_COLUMN, DataSchema.code_name).alias(MAPPED_CODE_COLUMN),
            (pl.col(DataSchema.code_name) == pl.coalesce(MAPPED_CODE_COLUMN, DataSchema.code_name)).alias(is_exact_match),
        )
        .sort(MAPPED_CODE_COLUMN, is_exact_match, DataSchema.code_name, descending=[False, True, False])
    )

    technical = {
        DataSchema.code_name,
        is_exact_match,
        MAPPED_CODE_COLUMN,
        COUNT_COLUMN,
        MAPPED_COUNT_COLUMN,
        PROFILE_COLUMN,
        REASON_COLUMN,
        MEMBER_COUNT_COLUMN,
    }
    preserved = [column for column in metadata.columns if column not in technical]
    aggregations = [pl.col(column).drop_nulls().first() for column in preserved]
    if COUNT_COLUMN in metadata.columns:
        aggregations.append(pl.col(COUNT_COLUMN).fill_null(0).sum().alias(COUNT_COLUMN))
    aggregations.append(pl.len().cast(pl.UInt32).alias(MEMBER_COUNT_COLUMN))

    collapsed = (
        mapped.group_by(MAPPED_CODE_COLUMN, maintain_order=True)
        .agg(*aggregations)
        .rename({MAPPED_CODE_COLUMN: DataSchema.code_name})
    )

    if "description" in collapsed.columns:
        generic = pl.format("Adaptive aggregation {} ({} source codes)", DataSchema.code_name, MEMBER_COUNT_COLUMN)
        collapsed = collapsed.with_columns(
            pl.when(pl.col(MEMBER_COUNT_COLUMN) > 1).then(generic).otherwise(pl.col("description")).alias("description")
        )
    if "parent_codes" in collapsed.columns:
        collapsed = collapsed.with_columns(
            pl.when(pl.col(MEMBER_COUNT_COLUMN) > 1)
            .then(pl.lit(None, dtype=pl.List(pl.String)))
            .otherwise(pl.col("parent_codes"))
            .alias("parent_codes")
        )
    return collapsed.sort(DataSchema.code_name)


def add_missing_observed_metadata(metadata: pl.DataFrame, observed_codes: Sequence[str]) -> pl.DataFrame:
    """Ensure finalized metadata covers every code present in transformed data."""
    missing_codes = sorted(set(observed_codes) - set(metadata.get_column(DataSchema.code_name).to_list()))
    if not missing_codes:
        return metadata

    columns: dict[str, pl.Series] = {}
    for name, dtype in metadata.schema.items():
        if name == DataSchema.code_name:
            columns[name] = pl.Series(name, missing_codes, dtype=pl.String)
        elif name == COUNT_COLUMN:
            columns[name] = pl.Series(name, [0] * len(missing_codes), dtype=dtype)
        elif name == MEMBER_COUNT_COLUMN:
            columns[name] = pl.Series(name, [1] * len(missing_codes), dtype=dtype)
        else:
            columns[name] = pl.Series(name, [None] * len(missing_codes), dtype=dtype)
    return pl.concat([metadata, pl.DataFrame(columns)]).sort(DataSchema.code_name)


@Stage.register(
    is_metadata=True,
    default_config=Path("configs/MEDS/default_adaptive_code_mapping.yaml"),
)
def main(cfg: DictConfig) -> None:
    """Collapse ``codes.parquet`` using the fitted or external mapping."""
    if cfg.worker != 0:
        return

    input_filepath = Path(str(cfg.stage_cfg.metadata_input_dir)) / "codes.parquet"
    if not input_filepath.is_file():
        raise FileNotFoundError(f"Adaptive code metadata input does not exist: {input_filepath}")
    metadata = pl.read_parquet(input_filepath)
    mapping = prepare_mapping(metadata, external_mapping_filepath=cfg.stage_cfg.get("mapping_filepath"))
    collapsed = collapse_code_metadata(metadata, mapping)
    data_input_dir = Path(str(cfg.stage_cfg.data_input_dir))
    data_files = sorted(data_input_dir.glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No transformed MEDS data shards found in {data_input_dir}")
    observed_codes = (
        pl.concat([pl.scan_parquet(path).select(DataSchema.code_name) for path in data_files])
        .select(pl.col(DataSchema.code_name).unique())
        .collect()
        .get_column(DataSchema.code_name)
        .to_list()
    )
    collapsed = add_missing_observed_metadata(collapsed, observed_codes)

    output_filepath = Path(str(cfg.stage_cfg.reducer_output_dir)) / "codes.parquet"
    if output_filepath.exists() and not cfg.do_overwrite:
        raise FileExistsError(f"Output file already exists: {output_filepath}")
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    collapsed.write_parquet(output_filepath)


stage = main
