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


def prepare_mapping(
    local_metadata: pl.DataFrame,
    external_mapping_filepath: str | None,
    external_mapping_mode: str,
) -> pl.DataFrame:
    """Select a local mapping and optionally overlay or replace it externally."""
    required_local = {DataSchema.code_name, MAPPED_CODE_COLUMN}
    if not required_local.issubset(local_metadata.columns):
        if external_mapping_filepath:
            local = pl.DataFrame(schema={DataSchema.code_name: pl.String, MAPPED_CODE_COLUMN: pl.String})
        else:
            missing = sorted(required_local - set(local_metadata.columns))
            raise ValueError(f"local adaptive metadata is missing columns: {missing}")
    else:
        local = local_metadata.select(DataSchema.code_name, MAPPED_CODE_COLUMN)

    if not external_mapping_filepath:
        return local
    external = load_frame(str(external_mapping_filepath), "external mapping")
    missing = required_local - set(external.columns)
    if missing:
        raise ValueError(f"external mapping is missing columns: {sorted(missing)}")
    external = external.select(DataSchema.code_name, MAPPED_CODE_COLUMN)
    if external.get_column(DataSchema.code_name).n_unique() != external.height:
        raise ValueError("external mapping must contain at most one row per code")

    mode = str(external_mapping_mode).lower()
    if mode == "replace":
        return external
    if mode != "overlay":
        raise ValueError("external_mapping_mode must be 'overlay' or 'replace'")
    return pl.concat([external, local]).unique(DataSchema.code_name, keep="first", maintain_order=True)
