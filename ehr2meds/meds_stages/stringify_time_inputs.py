"""Stringify preMEDS columns used as inputs to event time expressions.

NOTE; This might be a bit excessive. It could be that this can be solved
more cleanly upstream instead of as a custom stage.

However, MEDS_extract currently checks referenced time-source columns for
both null and empty-string values. It therefore gives errors for datetime
and integers.

This stage reads the MEDS event configuration, identifies which
columns are referenced by `time` expressions, and converts those
columns to strings.
"""

from __future__ import annotations

import logging
import polars as pl
import re
from collections.abc import Iterable
from dftly import extract_columns
from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

logger = logging.getLogger(__name__)


# These could potentially be moved to resources or constants etc
EVENT_META_KEYS = {
    "subject_id_col",
    "subject_id_expr",
    "transforms",
    "join",
}

DIRECT_FORMAT_CAST_PATTERN = re.compile(
    r"""\$(?P<column>[A-Za-z_]\w*)\s+as\s+"""
    r"""(?P<quote>["'])(?P<format>.*?)(?P=quote)"""
)

DIRECT_INTEGER_CAST_PATTERN = re.compile(
    r"""\$(?P<column>[A-Za-z_]\w*)\s+as\s+"""
    r"""(?P<cast>int|integer|int8|int16|int32|int64|int128|long|"""
    r"""uint|uint8|uint16|uint32|uint64)\b"""
)

INTEGER_CASTS = {
    "int",
    "integer",
    "int8",
    "int16",
    "int32",
    "int64",
    "int128",
    "long",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}

DATE_FORMAT_PARTS = {
    "%Y",
    "%y",
    "%m",
    "%b",
    "%B",
    "%d",
    "%e",
    "%j",
    "%F",
}

TIME_FORMAT_PARTS = {
    "%H",
    "%k",
    "%I",
    "%l",
    "%M",
    "%S",
    "%f",
    "%.f",
    "%3f",
    "%6f",
    "%9f",
    "%p",
    "%R",
    "%T",
}


def get_temporal_format_type(format_string: str) -> str:
    """Infer whether a strptime format describes a date, time, or datetime."""

    has_date = any(part in format_string for part in DATE_FORMAT_PARTS)
    has_time = any(part in format_string for part in TIME_FORMAT_PARTS)

    if has_date and has_time:
        return "datetime"

    if has_date:
        return "date"

    if has_time:
        return "time"

    raise ValueError(f"Could not infer temporal type from format {format_string!r}")


def normalize_temporal_string(
    source: pl.Expr,
    format_string: str,
) -> pl.Expr:
    """Normalize an existing temporal string to a configured format.

    Parsing is handles representations such as:

        9:59:00.000000
        09:59:00
        2008-08-08 00:00:00
        2008-08-08

    If no parser matches, the original string is preserved.
    """

    text = source.cast(pl.String).str.strip_chars()
    is_missing = text.is_null() | (text == "")

    temporal_type = get_temporal_format_type(format_string)

    if temporal_type == "time":
        parsed = pl.coalesce(
            # Already matches the requested representation.
            text.str.strptime(
                pl.Time,
                format=format_string,
                strict=False,
            ),
            # Common Arrow/pandas representation with fractional seconds.
            text.str.strptime(
                pl.Time,
                format="%H:%M:%S%.f",
                strict=False,
            ),
            # Let Polars infer other valid time representations.
            text.str.strptime(
                pl.Time,
                strict=False,
            ),
        )

    elif temporal_type == "date":
        parsed = pl.coalesce(
            text.str.strptime(
                pl.Date,
                format=format_string,
                strict=False,
            ),
            text.str.strptime(
                pl.Date,
                strict=False,
            ),
            # Handles strings such as "2008-08-08 00:00:00".
            text.str.strptime(
                pl.Datetime,
                strict=False,
            ).cast(pl.Date),
        )

    else:
        parsed = pl.coalesce(
            text.str.strptime(
                pl.Datetime,
                format=format_string,
                strict=False,
            ),
            text.str.strptime(
                pl.Datetime,
                strict=False,
            ),
        )

    canonical = parsed.dt.strftime(format_string)

    return (
        pl.when(is_missing)
        .then(pl.lit(None, dtype=pl.String))
        # Keep the original value if parsing failed, so errors remain visible.
        .otherwise(pl.coalesce(canonical, text))
    )


def get_time_specs_by_source(
    event_config_fp: str | Path,
) -> dict[str, dict[str, str | None]]:
    """Find columns and direct casts used inside event time expressions."""

    event_config_fp = Path(event_config_fp)

    if not event_config_fp.exists():
        raise FileNotFoundError(f"Event conversion config does not exist: {event_config_fp}")

    config = OmegaConf.to_container(
        OmegaConf.load(event_config_fp),
        resolve=True,
    )

    if not isinstance(config, dict):
        raise TypeError("The event conversion configuration must resolve to a dictionary.")

    config.pop("subject_id_col", None)

    result: dict[str, dict[str, str | None]] = {}

    for source_name, source_config in config.items():
        if not isinstance(source_config, dict):
            continue

        time_specs: dict[str, str | None] = {}

        for event_name, event_config in source_config.items():
            if event_name in EVENT_META_KEYS:
                continue

            if not isinstance(event_config, dict):
                continue

            time_expression = event_config.get("time")

            if not isinstance(time_expression, str):
                continue

            referenced_columns = extract_columns(time_expression)

            direct_specs: dict[str, str] = {}

            for match in DIRECT_FORMAT_CAST_PATTERN.finditer(time_expression):
                direct_specs[match.group("column")] = match.group("format")

            # Note; sometimes we need to find these patterns
            # inside f-strings, with examples being instances
            # like:
            #
            #   {$v_otime as int}
            #   {$v_ominut as int}
            for match in DIRECT_INTEGER_CAST_PATTERN.finditer(time_expression):
                direct_specs[match.group("column")] = match.group("cast")

            for column in referenced_columns:
                requested_spec = direct_specs.get(column)
                previous_spec = time_specs.get(column)

                if previous_spec is not None and requested_spec is not None and previous_spec != requested_spec:
                    raise ValueError(
                        f"Column {column!r} in source "
                        f"{source_name!r} is used with conflicting "
                        f"time specifications: {previous_spec!r} and "
                        f"{requested_spec!r}"
                    )

                if requested_spec is not None:
                    time_specs[column] = requested_spec
                elif column not in time_specs:
                    time_specs[column] = None

        result[str(source_name)] = time_specs

    return result


def _is_datetime_dtype(dtype: pl.DataType) -> bool:
    base_type = getattr(dtype, "base_type", None)

    if callable(base_type):
        return base_type() == pl.Datetime

    return dtype == pl.Datetime


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    is_numeric = getattr(dtype, "is_numeric", None)
    return bool(is_numeric()) if callable(is_numeric) else False


def stringify_time_input_columns(
    df: pl.LazyFrame,
    time_specs: dict[str, str | None],
) -> pl.LazyFrame:
    """Convert time-expression inputs to their configured string formats."""

    if not time_specs:
        return df

    schema = df.collect_schema()
    schema_names = set(schema.names())

    missing_columns = sorted(set(time_specs) - schema_names)

    if missing_columns:
        raise ValueError(
            "Configured time-input columns are missing from this source: "
            f"{missing_columns}. Available columns: {sorted(schema_names)}"
        )

    expressions: list[pl.Expr] = []

    for column, requested_spec in sorted(time_specs.items()):
        dtype = schema[column]
        source = pl.col(column)

        # Normalize 10, 10.0, "10", and "10.0" to "10".
        if requested_spec in INTEGER_CASTS:
            text = source.cast(pl.String).str.strip_chars()

            expression = (
                pl.when(text.is_null() | (text == ""))
                .then(pl.lit(None, dtype=pl.String))
                .otherwise(text.cast(pl.Float64, strict=True).cast(pl.Int64).cast(pl.String))
            )

        elif isinstance(requested_spec, str) and "%" in requested_spec:
            if dtype == pl.String:
                expression = normalize_temporal_string(
                    source,
                    format_string=requested_spec,
                )

            elif dtype == pl.Date:
                expression = source.dt.strftime(requested_spec)

            elif _is_datetime_dtype(dtype):
                expression = source.dt.strftime(requested_spec)

            elif dtype == pl.Time:
                expression = source.dt.strftime(requested_spec)

            else:
                expression = source.cast(pl.String)

        elif dtype == pl.String:
            continue

        elif dtype == pl.Date:
            expression = source.dt.strftime("%Y-%m-%d")

        elif _is_datetime_dtype(dtype):
            expression = source.dt.strftime("%Y-%m-%d %H:%M:%S")

        elif dtype == pl.Time:
            expression = source.dt.strftime("%H:%M:%S")

        elif _is_numeric_dtype(dtype):
            expression = source.cast(pl.String)

        else:
            expression = source.cast(pl.String)

        expressions.append(expression.alias(column))

    if not expressions:
        return df

    return df.with_columns(expressions)


def identify_source_name(
    input_fp: Path,
    input_dir: Path,
    source_names: Iterable[str],
) -> str:
    """Match a subject-sharded file to its event-config source name.

    The source name is matched against the final path components.
    """
    relative_without_suffix = input_fp.relative_to(input_dir).with_suffix("")
    relative_parts = relative_without_suffix.parts

    ordered_sources = sorted(
        source_names,
        key=lambda name: len(Path(name).parts),
        reverse=True,
    )

    for source_name in ordered_sources:
        source_parts = Path(source_name).parts

        if len(source_parts) <= len(relative_parts) and tuple(relative_parts[-len(source_parts) :]) == source_parts:
            return source_name

    raise ValueError(f"Could not match input file {input_fp} to an event-config source. Known sources: {sorted(source_names)}")


@Stage.register(is_metadata=False)
def main(cfg: DictConfig) -> None:
    """Stringify configured time-input columns in subject-sharded preMEDS."""
    input_dir = Path(str(cfg.stage_cfg.data_input_dir))
    output_dir = Path(str(cfg.stage_cfg.output_dir))
    event_config_fp = Path(str(cfg.event_conversion_config_fp))

    if not input_dir.exists():
        raise FileNotFoundError(f"Stage input directory does not exist: {input_dir}")

    time_specs_by_source = get_time_specs_by_source(event_config_fp)

    logger.info(
        "Detected time-input specifications:\n%s",
        "\n".join(f"  {source}: {specs}" for source, specs in sorted(time_specs_by_source.items())),
    )

    input_files = sorted(input_dir.rglob("*.parquet"))

    if not input_files:
        raise FileNotFoundError(f"No parquet files found under stage input directory: {input_dir}")

    do_overwrite = bool(cfg.get("do_overwrite", False))

    for input_fp in input_files:
        source_name = identify_source_name(
            input_fp=input_fp,
            input_dir=input_dir,
            source_names=time_specs_by_source.keys(),
        )

        time_specs = time_specs_by_source[source_name]

        output_fp = output_dir / input_fp.relative_to(input_dir)

        logger.info(
            "Processing %s as source %s. Time-input specifications: %s",
            input_fp,
            source_name,
            time_specs,
        )

        def compute_fn(
            df: pl.LazyFrame,
            specs: dict[str, str | None] = time_specs,
        ) -> pl.LazyFrame:
            return stringify_time_input_columns(
                df,
                time_specs=specs,
            )

        rwlock_wrap(
            input_fp,
            output_fp,
            read_df,
            write_df,
            compute_fn,
            do_overwrite=do_overwrite,
        )
