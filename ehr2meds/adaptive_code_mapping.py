"""Column names and mapping I/O shared by the adaptive code-mapping stages."""

from __future__ import annotations

import polars as pl
from meds import DataSchema
from MEDS_transforms.utils import PKG_PFX, resolve_pkg_path
from pathlib import Path

CODE_COLUMN = DataSchema.code_name
MAPPED_CODE_COLUMN = "adaptive/mapped_code"
COUNT_COLUMN = "adaptive/count"
MAPPED_COUNT_COLUMN = "adaptive/mapped_count"
PROFILE_COLUMN = "adaptive/profile"
REASON_COLUMN = "adaptive/reason"
MEMBER_COUNT_COLUMN = "adaptive/member_count"


def resolve_resource_path(filepath: str) -> Path:
    """Resolve normal and package resource paths."""
    return resolve_pkg_path(filepath) if filepath.startswith(PKG_PFX) else Path(filepath)


def load_frame(filepath: str, label: str) -> pl.DataFrame:
    """Load a JSON or Parquet mapping resource."""
    path = resolve_resource_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"{label} filepath '{filepath}' does not exist")
    match path.suffix.lower():
        case ".parquet":
            return pl.read_parquet(path)
        case ".json":
            return pl.read_json(path)
        case _:
            raise ValueError(f"{label} filepath must point to a JSON or Parquet file")


def prepare_mapping(
    local_metadata: pl.DataFrame,
    external_mapping_filepath: str | None,
    external_mapping_mode: str,
) -> pl.DataFrame:
    """Select a local mapping and optionally overlay or replace it externally."""
    required_local = {CODE_COLUMN, MAPPED_CODE_COLUMN}
    if not required_local.issubset(local_metadata.columns):
        if external_mapping_filepath:
            local = pl.DataFrame(schema={CODE_COLUMN: pl.String, MAPPED_CODE_COLUMN: pl.String})
        else:
            missing = sorted(required_local - set(local_metadata.columns))
            raise ValueError(f"local adaptive metadata is missing columns: {missing}")
    else:
        local = local_metadata.select(CODE_COLUMN, MAPPED_CODE_COLUMN)

    if not external_mapping_filepath:
        return local
    external = load_frame(str(external_mapping_filepath), "external mapping")
    missing = required_local - set(external.columns)
    if missing:
        raise ValueError(f"external mapping is missing columns: {sorted(missing)}")
    external = external.select(CODE_COLUMN, MAPPED_CODE_COLUMN)
    if external.get_column(CODE_COLUMN).n_unique() != external.height:
        raise ValueError("external mapping must contain at most one row per code")

    mode = str(external_mapping_mode).lower()
    if mode == "replace":
        return external
    if mode != "overlay":
        raise ValueError("external_mapping_mode must be 'overlay' or 'replace'")
    return pl.concat([external, local]).unique(CODE_COLUMN, keep="first", maintain_order=True)
