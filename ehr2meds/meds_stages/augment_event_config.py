"""Generate a MEDS event config with shared columns added to every event."""

from __future__ import annotations

from collections.abc import Mapping
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

DEFAULT_EVENT_COLUMNS = {"row_idx": "$row_idx"}


def _read_config(path: Path) -> DictConfig:
    """Read an event configuration and require a YAML mapping at its root."""
    if not path.is_file():
        raise FileNotFoundError(f"Event configuration does not exist: {path}")

    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise TypeError("Event configuration must contain a top-level mapping")
    return config


def _validate_event_columns(value: object) -> dict[str, str]:
    """Return event columns as a plain mapping after validating the config."""
    if not isinstance(value, Mapping) or not all(
        isinstance(column, str) and isinstance(expression, str) for column, expression in value.items()
    ):
        raise TypeError("event_columns must map output column names to dftly expression strings")
    if not value:
        raise ValueError("event_columns must contain at least one column")
    return dict(value)


def _write_config(config: DictConfig, path: Path) -> None:
    """Write readable YAML while preserving the original key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path, resolve=False)


def _add_event_columns(config: DictConfig, columns: Mapping[str, str]) -> int:
    """Add shared columns to every event and return the number of events found."""
    event_count = 0

    for file_name, file_config in config.items():
        if not isinstance(file_config, DictConfig):
            continue

        for event_name, event_config in file_config.items():
            # Using code as a proxy for event definitions, skipping other blocks
            if not isinstance(event_config, DictConfig) or "code" not in event_config:
                continue

            event_count += 1
            for column_name, expression in columns.items():
                existing_expression = event_config.get(column_name)
                if existing_expression is not None and existing_expression != expression:
                    field = f"{file_name}.{event_name}.{column_name}"
                    raise ValueError(
                        f"Event field {field!r} is already {existing_expression!r}; cannot set it to {expression!r}"
                    )
                event_config[column_name] = expression

    return event_count


def augment_event_config(
    src_fp: Path,
    out_fp: Path,
    *,
    event_columns: Mapping[str, str] = DEFAULT_EVENT_COLUMNS,
) -> Path:
    """Add shared output columns to every event definition.

    Event blocks are identified by the presence of ``code``. This skips
    structural blocks such as ``subject_id_col``, ``transforms``, and ``join``.
    Existing identical declarations are accepted; conflicts raise an error.
    """
    columns = _validate_event_columns(event_columns)
    config = _read_config(src_fp)

    event_count = _add_event_columns(config, columns)
    if event_count == 0:
        raise ValueError(f"Event configuration contains no event definitions: {src_fp}")

    _write_config(config, out_fp)
    return out_fp


@Stage.register(is_metadata=False)
def main(cfg: DictConfig) -> None:
    """Create the augmented config consumed by subsequent extraction stages."""
    stage_cfg = cfg.stage_cfg
    event_columns = stage_cfg.get("event_columns", DEFAULT_EVENT_COLUMNS)

    augment_event_config(
        Path(str(stage_cfg.source_event_conversion_config_fp)),
        Path(str(stage_cfg.output_event_conversion_config_fp)),
        event_columns=event_columns,
    )
