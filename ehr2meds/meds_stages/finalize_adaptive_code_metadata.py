"""Rewrite MEDS code metadata to the adaptively mapped vocabulary."""

from __future__ import annotations

import polars as pl
from collections.abc import Mapping, Sequence
from ehr2meds.adaptive_code_mapping import prepare_mapping
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def collapse_code_metadata(
    metadata: pl.DataFrame,
    mapping: pl.DataFrame,
    columns: Mapping[str, str],
) -> pl.DataFrame:
    """Rewrite and deterministically collapse code metadata."""
    exact_match_column = "is_exact_match"
    member_count_column = "member_count"
    mapped_code_column = columns["mapped_code"]

    mapped_code = pl.coalesce(
        pl.col(DataSchema.code_name).replace(
            old=mapping.get_column(DataSchema.code_name),
            new=mapping.get_column(mapped_code_column),
        ),
        pl.col(DataSchema.code_name),
    )
    mapped = metadata.with_columns(
        mapped_code.alias(mapped_code_column),
        (pl.col(DataSchema.code_name) == mapped_code).alias(exact_match_column),
    )
    mapped = mapped.sort(
        mapped_code_column,
        exact_match_column,
        DataSchema.code_name,
        descending=[False, True, False],
    )

    technical_columns = {
        DataSchema.code_name,
        exact_match_column,
        member_count_column,
        *columns.values(),
    }
    preserved_columns = [column for column in metadata.columns if column not in technical_columns]
    aggregations = [pl.col(column).drop_nulls().first() for column in preserved_columns]
    aggregations.append(pl.len().cast(pl.UInt32).alias(member_count_column))

    collapsed = mapped.group_by(mapped_code_column, maintain_order=True).agg(aggregations)
    collapsed = collapsed.rename({mapped_code_column: DataSchema.code_name})

    if "description" in collapsed.columns:
        aggregate_description = pl.format(
            "Adaptive aggregation {} ({} source codes)", DataSchema.code_name, member_count_column
        )
        collapsed = collapsed.with_columns(
            description=pl.when(pl.col(member_count_column) > 1).then(aggregate_description).otherwise(pl.col("description"))
        )
    if "parent_codes" in collapsed.columns:
        collapsed = collapsed.with_columns(
            parent_codes=pl.when(pl.col(member_count_column) > 1)
            .then(pl.lit(None, dtype=pl.List(pl.String)))
            .otherwise(pl.col("parent_codes"))
        )
    collapsed = collapsed.drop(member_count_column)
    return collapsed.sort(DataSchema.code_name)


def add_missing_observed_metadata(
    metadata: pl.DataFrame,
    observed_codes: Sequence[str],
) -> pl.DataFrame:
    """Ensure finalized metadata covers every code present in transformed data."""
    missing_codes = sorted(set(observed_codes) - set(metadata.get_column(DataSchema.code_name).to_list()))
    if not missing_codes:
        return metadata

    null_values = [None] * len(missing_codes)
    missing_metadata = {
        name: pl.Series(name, missing_codes if name == DataSchema.code_name else null_values, dtype=dtype)
        for name, dtype in metadata.schema.items()
    }
    metadata = pl.concat([metadata, pl.DataFrame(missing_metadata)])
    return metadata.sort(DataSchema.code_name)


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
    columns = cfg.stage_cfg.columns
    mapping = prepare_mapping(
        metadata,
        external_mapping_filepath=cfg.stage_cfg.get("mapping_filepath"),
        columns=columns,
    )
    collapsed = collapse_code_metadata(metadata, mapping, columns)
    data_input_dir = Path(str(cfg.stage_cfg.data_input_dir))
    data_files = sorted(data_input_dir.glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No transformed MEDS data shards found in {data_input_dir}")
    observed = pl.concat([pl.scan_parquet(path).select(DataSchema.code_name) for path in data_files])
    observed = observed.select(pl.col(DataSchema.code_name).unique()).collect()
    observed_codes = observed.get_column(DataSchema.code_name).to_list()
    collapsed = add_missing_observed_metadata(collapsed, observed_codes)

    output_filepath = Path(str(cfg.stage_cfg.reducer_output_dir)) / "codes.parquet"
    if output_filepath.exists() and not cfg.do_overwrite:
        raise FileExistsError(f"Output file already exists: {output_filepath}")
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    collapsed.write_parquet(output_filepath)


stage = main
