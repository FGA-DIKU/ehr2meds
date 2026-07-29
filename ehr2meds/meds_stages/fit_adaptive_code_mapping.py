"""Fit adaptive hierarchical code mappings on training event counts."""

from __future__ import annotations

import json
import polars as pl
from collections.abc import Callable
from ehr2meds.meds_stages.adaptive_code_mapping import (
    COUNT_COLUMN,
    add_unseen_metadata_codes,
    combine_count_frames,
    fit_mapping,
    read_profiles,
    summarize_mapping,
)
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


def mapper_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Count training events per code; ``train_only`` selects the shards."""
    read_profiles(stage_cfg)

    def mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.group_by(DataSchema.code_name)
            .len()
            .select(
                pl.col(DataSchema.code_name),
                pl.col("len").cast(pl.UInt64).alias(COUNT_COLUMN),
            )
            .sort(DataSchema.code_name)
        )

    return mapper


def reducer_fntr(stage_cfg: DictConfig) -> Callable[..., pl.LazyFrame]:
    """Fit one global mapping and write its reusable mapping and audit files."""
    profiles, namespaces = read_profiles(stage_cfg)
    configured_output = stage_cfg.get("mapping_output_filepath")
    output_filepath = (
        Path(str(configured_output))
        if configured_output
        else Path(str(stage_cfg.reducer_output_dir)) / "adaptive_code_mapping.parquet"
    )
    configured_summary = stage_cfg.get("mapping_summary_output_filepath")
    if configured_summary:
        summary_filepath = Path(str(configured_summary))
    else:
        summary_filepath = output_filepath.with_suffix(".summary.json")
    code_metadata_filepath = Path(str(stage_cfg.metadata_input_dir)) / "codes.parquet"
    code_metadata = (
        pl.read_parquet(code_metadata_filepath)
        if code_metadata_filepath.is_file()
        else pl.DataFrame(schema={DataSchema.code_name: pl.String})
    )

    def reducer(*dfs: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        counts = combine_count_frames(*dfs)
        mapping = fit_mapping(counts, profiles=profiles, namespaces=namespaces)
        mapping = add_unseen_metadata_codes(mapping, code_metadata)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        mapping.write_parquet(output_filepath)
        summary_filepath.parent.mkdir(parents=True, exist_ok=True)
        summary_filepath.write_text(
            json.dumps(summarize_mapping(mapping), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        # MEDS-Transforms scans prior code metadata lazily, and its merge helper
        # requires both sides of the join to use the same eager/lazy type.
        return mapping.lazy()

    return reducer


stage = Stage.register(
    map_fn=mapper_fntr,
    reduce_fn=reducer_fntr,
    default_config=Path("configs/MEDS/default_adaptive_code_mapping.yaml"),
)
