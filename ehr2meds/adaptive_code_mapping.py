"""Column names and mapping I/O shared by the adaptive code-mapping stages."""

from __future__ import annotations

import polars as pl
from ehr2meds.io_utils import load_frame
from meds import DataSchema

MAPPED_CODE_COLUMN = "adaptive/mapped_code"
COUNT_COLUMN = "adaptive/count"
MAPPED_COUNT_COLUMN = "adaptive/mapped_count"
PROFILE_COLUMN = "adaptive/profile"
REASON_COLUMN = "adaptive/reason"
MEMBER_COUNT_COLUMN = "adaptive/member_count"


def prepare_mapping(local_metadata: pl.DataFrame, external_mapping_filepath: str | None) -> pl.DataFrame:
    """Use the local fitted mapping if one exists; otherwise fall back to an external mapping."""
    required_local = {DataSchema.code_name, MAPPED_CODE_COLUMN}
    if required_local.issubset(local_metadata.columns):
        return local_metadata.select(DataSchema.code_name, MAPPED_CODE_COLUMN)

    if not external_mapping_filepath:
        missing = sorted(required_local - set(local_metadata.columns))
        raise ValueError(f"local adaptive metadata is missing columns: {missing}")
    external = load_frame(str(external_mapping_filepath), "external mapping")
    missing = required_local - set(external.columns)
    if missing:
        raise ValueError(f"external mapping is missing columns: {sorted(missing)}")
    external = external.select(DataSchema.code_name, MAPPED_CODE_COLUMN)
    if external.get_column(DataSchema.code_name).n_unique() != external.height:
        raise ValueError("external mapping must contain at most one row per code")
    return external
