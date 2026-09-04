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
    is_exact_match = "is_exact_match"
    mapped_code_column = columns["mapped_code"]
    count_column = columns["count"]
    member_count_column = columns["member_count"]
    mapped_code = pl.coalesce(
        pl.col(DataSchema.code_name).replace(
            old=mapping[DataSchema.code_name],
            new=mapping[mapped_code_column],
        ),
        pl.col(DataSchema.code_name),
    )
    mapped = metadata.with_columns(
        **{
            mapped_code_column: mapped_code,
            is_exact_match: pl.col(DataSchema.code_name) == mapped_code,
        }
    ).sort(mapped_code_column, is_exact_match, DataSchema.code_name, descending=[False, True, False])

    technical = {
        DataSchema.code_name,
        is_exact_match,
        *columns.values(),
    }
    preserved = [column for column in metadata.columns if column not in technical]
    aggregations = {column: pl.col(column).drop_nulls().first() for column in preserved}
    if count_column in metadata.columns:
        aggregations[count_column] = pl.col(count_column).fill_null(0).sum()
    aggregations[member_count_column] = pl.len().cast(pl.UInt32)

    collapsed = (
        mapped.group_by(mapped_code_column, maintain_order=True)
        .agg(**aggregations)
        .rename({mapped_code_column: DataSchema.code_name})
    )

    if "description" in collapsed.columns:
        generic = pl.format("Adaptive aggregation {} ({} source codes)", DataSchema.code_name, member_count_column)
        collapsed = collapsed.with_columns(
            description=pl.when(pl.col(member_count_column) > 1).then(generic).otherwise(pl.col("description"))
        )
    if "parent_codes" in collapsed.columns:
        collapsed = collapsed.with_columns(
            parent_codes=pl.when(pl.col(member_count_column) > 1)
            .then(pl.lit(None, dtype=pl.List(pl.String)))
            .otherwise(pl.col("parent_codes"))
        )
    return collapsed.sort(DataSchema.code_name)


def add_missing_observed_metadata(
    metadata: pl.DataFrame,
    observed_codes: Sequence[str],
    columns: Mapping[str, str],
) -> pl.DataFrame:
    """Ensure finalized metadata covers every code present in transformed data."""
    missing_codes = sorted(set(observed_codes) - set(metadata.get_column(DataSchema.code_name).to_list()))
    if not missing_codes:
        return metadata

    count_column = columns["count"]
    member_count_column = columns["member_count"]
    special_values = {
        DataSchema.code_name: missing_codes,
        member_count_column: [1] * len(missing_codes),
    }
    if count_column in metadata.columns:
        special_values[count_column] = [0] * len(missing_codes)

    null_values = [None] * len(missing_codes)
    output_columns = {
        name: pl.Series(name, special_values.get(name, null_values), dtype=dtype) for name, dtype in metadata.schema.items()
    }
    return pl.concat([metadata, pl.DataFrame(output_columns)]).sort(DataSchema.code_name)


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
    observed_codes = (
        pl.concat([pl.scan_parquet(path).select(DataSchema.code_name) for path in data_files])
        .select(pl.col(DataSchema.code_name).unique())
        .collect()
        .get_column(DataSchema.code_name)
        .to_list()
    )
    collapsed = add_missing_observed_metadata(collapsed, observed_codes, columns)

    output_filepath = Path(str(cfg.stage_cfg.reducer_output_dir)) / "codes.parquet"
    if output_filepath.exists() and not cfg.do_overwrite:
        raise FileExistsError(f"Output file already exists: {output_filepath}")
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    collapsed.write_parquet(output_filepath)


stage = main
