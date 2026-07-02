"""Add stable within-file row identifiers to preMEDS source tables.
Currently, we're doing data_source:row_id. Might be excessive, but
thought it could be useful to be able to group cleanly on just
that column in BONSAI"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig
import polars as pl

logger = logging.getLogger(__name__)


ROW_INDEX_COLUMN = "source_row_index"
ROW_ID_COLUMN = "source_row_id"


def add_source_row_id(
    df: pl.LazyFrame,
    *,
    source_name: str,
) -> pl.LazyFrame:
    return df.with_row_index("source_row_index").with_columns(
        pl.concat_str(
            [
                pl.lit(source_name),
                pl.col("source_row_index").cast(pl.String),
            ],
            separator=":",
        ).alias("source_row_id")
    )


@Stage.register(is_metadata=False)
def main(cfg: DictConfig) -> None:
    """Add source-row identifiers before MEDS extraction and sharding."""

    input_dir = Path(str(cfg.stage_cfg.data_input_dir))
    output_dir = Path(str(cfg.stage_cfg.output_dir))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_files = sorted(input_dir.rglob("*.parquet"))

    if not input_files:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")

    do_overwrite = bool(cfg.get("do_overwrite", False))

    for input_fp in input_files:
        relative_fp = input_fp.relative_to(input_dir)
        source_name = relative_fp.with_suffix("").as_posix()

        output_fp = output_dir / relative_fp

        logger.info(
            "Adding source-row identifiers to %s using source name %s",
            input_fp,
            source_name,
        )

        compute_fn = partial(
            add_source_row_id,
            source_name=source_name,
        )

        rwlock_wrap(
            input_fp,
            output_fp,
            read_df,
            write_df,
            compute_fn,
            do_overwrite=do_overwrite,
        )
