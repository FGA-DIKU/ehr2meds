"""Shared file-resolution and loading helpers used across ehr2meds stages."""

from __future__ import annotations

import polars as pl
from MEDS_transforms.utils import PKG_PFX, resolve_pkg_path
from pathlib import Path


def resolve_resource_path(filepath: str) -> Path:
    """Resolve normal and package resource paths."""
    return resolve_pkg_path(filepath) if filepath.startswith(PKG_PFX) else Path(filepath)


def load_frame(filepath: str, label: str) -> pl.DataFrame:
    """Load a JSON or Parquet resource."""
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
