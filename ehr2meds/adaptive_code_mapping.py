"""Mapping I/O shared by the adaptive code-mapping stages."""

from __future__ import annotations

import polars as pl
from collections.abc import Mapping
from ehr2meds.io_utils import load_frame
from meds import DataSchema


def prepare_mapping(
    local_metadata: pl.DataFrame,
    external_mapping_filepath: str | None,
    columns: Mapping[str, str],
) -> pl.DataFrame:
    """Use the local fitted mapping if one exists; otherwise fall back to an external mapping."""
    mapped_code_column = columns["mapped_code"]
    required_local = {DataSchema.code_name, mapped_code_column}
    if required_local.issubset(local_metadata.columns):
        local = local_metadata.select(DataSchema.code_name, mapped_code_column)
        if local.get_column(DataSchema.code_name).n_unique() != local.height:
            raise ValueError("local adaptive mapping must contain at most one row per code")
        return local

    if not external_mapping_filepath:
        missing = sorted(required_local - set(local_metadata.columns))
        raise ValueError(f"local adaptive metadata is missing columns: {missing}")
    external = load_frame(str(external_mapping_filepath), "external mapping")
    missing = required_local - set(external.columns)
    if missing:
        raise ValueError(f"external mapping is missing columns: {sorted(missing)}")
    external = external.select(DataSchema.code_name, mapped_code_column)
    if external.get_column(DataSchema.code_name).n_unique() != external.height:
        raise ValueError("external mapping must contain at most one row per code")
    return external
