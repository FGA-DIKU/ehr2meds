"""Rewrite MEDS code metadata to the adaptively mapped vocabulary."""

from __future__ import annotations

import polars as pl
from ehr2meds.meds_stages.adaptive_code_mapping import (
    CODE_COLUMN,
    add_missing_observed_metadata,
    collapse_code_metadata,
    prepare_mapping,
)
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
from pathlib import Path


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
    mapping = prepare_mapping(
        metadata,
        external_mapping_filepath=cfg.stage_cfg.get("mapping_filepath"),
        external_mapping_mode=str(cfg.stage_cfg.get("external_mapping_mode", "overlay")),
    )
    collapsed = collapse_code_metadata(metadata, mapping)
    data_input_dir = Path(str(cfg.stage_cfg.data_input_dir))
    data_files = sorted(data_input_dir.glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No transformed MEDS data shards found in {data_input_dir}")
    observed_codes = (
        pl.concat([pl.scan_parquet(path).select(CODE_COLUMN) for path in data_files])
        .select(pl.col(CODE_COLUMN).unique())
        .collect()
        .get_column(CODE_COLUMN)
        .to_list()
    )
    collapsed = add_missing_observed_metadata(collapsed, observed_codes)

    output_filepath = Path(str(cfg.stage_cfg.reducer_output_dir)) / "codes.parquet"
    if output_filepath.exists() and not cfg.do_overwrite:
        raise FileExistsError(f"Output file already exists: {output_filepath}")
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    collapsed.write_parquet(output_filepath)


stage = main
