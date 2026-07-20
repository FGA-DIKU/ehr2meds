"""Generate a MEDS event config with shared columns added to every event."""

from __future__ import annotations

import yaml
from collections.abc import Mapping
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

DEFAULT_EVENT_COLUMNS = {
    "source_row_id": "$source_row_id",
    "source_row_index": "$source_row_index",
}


def augment_event_config(
    src_fp: Path,
    out_fp: Path,
    *,
    event_columns: Mapping[str, str] = DEFAULT_EVENT_COLUMNS,
) -> Path:
    """Add shared output columns to every event definition.

    An event definition is identified structurally by the presence of a
    ``code`` field. This naturally skips global/per-file settings such as
    ``subject_id_col``, ``transforms``, ``schema``, and ``join``.

    Existing fields are accepted when they have the requested expression. A
    conflicting expression raises instead of being silently overwritten.
    """
    if not src_fp.is_file():
        raise FileNotFoundError(f"Event configuration does not exist: {src_fp}")

    config = yaml.safe_load(src_fp.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Event configuration must contain a top-level mapping")

    for file_block in config.values():
        if not isinstance(file_block, dict):
            continue
        for event_block in file_block.values():
            if not isinstance(event_block, dict) or "code" not in event_block:
                continue
            for column, expression in event_columns.items():
                existing = event_block.get(column)
                if existing is not None and existing != expression:
                    raise ValueError(
                        f"Event column {column!r} already has conflicting expression {existing!r}; requested {expression!r}"
                    )
                event_block[column] = expression

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    out_fp.write_text(rendered, encoding="utf-8")
    return out_fp


@Stage.register(is_metadata=False)
def main(cfg: DictConfig) -> None:
    """Pipeline entry point for generating the augmented event config."""
    stage_cfg = cfg.stage_cfg
    configured_columns = stage_cfg.get("event_columns", DEFAULT_EVENT_COLUMNS)
    event_columns = (
        OmegaConf.to_container(configured_columns) if OmegaConf.is_config(configured_columns) else configured_columns
    )
    if not isinstance(event_columns, dict) or not all(
        isinstance(column, str) and isinstance(expression, str) for column, expression in event_columns.items()
    ):
        raise TypeError("event_columns must map output column names to dftly expression strings")

    augment_event_config(
        Path(str(stage_cfg.source_event_conversion_config_fp)),
        Path(str(stage_cfg.output_event_conversion_config_fp)),
        event_columns=event_columns,
    )
