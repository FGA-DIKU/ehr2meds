"""Construction of complete timestamps from separate date and time columns."""

import logging
import polars as pl
from ehr2meds.preMEDS.utils import as_number

logger = logging.getLogger(__name__)

# Microseconds per unit and largest valid value, ordered from coarsest to finest.
TIME_COMPONENTS = {"hour": (3_600_000_000, 23), "minute": (60_000_000, 59), "second": (1_000_000, 59)}

# Tried in order when no format is configured. ISO and compact dates are year-first; separated
# dates are day-first (Danish convention). Each format only matches its own shape, so order is safe.
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d%b%Y")
TIME_FORMATS = ("%H:%M:%S%.f", "%H:%M")


def add_timestamp_columns(df: pl.DataFrame, timestamp_config: dict) -> pl.DataFrame:
    """Combine each date with as much valid time information as is available."""
    for output_column, config in timestamp_config.items():
        date = parse_date(df, config["date"], config.get("format"))
        warn_unparsed(df, config["date"], date)
        df = df.with_columns((date + create_time_delta(df, config)).cast(pl.Datetime("us")).alias(output_column))
    return df


def parse_date(df: pl.DataFrame, column: str, date_format: str | None) -> pl.Expr:
    """Parse a column to midnight of its date; unparseable values become null."""
    dtype = df.schema[column]
    if dtype == pl.Date or isinstance(dtype, pl.Datetime):
        return pl.col(column).cast(pl.Date).cast(pl.Datetime("us"))

    values = pl.col(column).cast(pl.String).str.strip_chars()
    if date_format:
        return values.str.to_datetime(date_format, strict=False, time_unit="us").dt.date().cast(pl.Datetime("us"))
    # Drop a trailing time of day ("2020-01-05 13:45", "...T13:45:00", SAS "05JAN2020:13:45:00").
    values = values.str.replace(r"[ T:]\d{1,2}:\d{2}.*$", "")
    return pl.coalesce(values.str.to_date(fmt, strict=False) for fmt in DATE_FORMATS).cast(pl.Datetime("us"))


def warn_unparsed(df: pl.DataFrame, column: str, date: pl.Expr) -> None:
    """Rows without a timestamp are dropped later, so say which values caused it."""
    failed = df.filter(pl.col(column).is_not_null() & date.is_null())[column]
    if len(failed):
        examples = failed.cast(pl.String).unique(maintain_order=True).head(5).to_list()
        logger.warning(f"{len(failed):_} values in {column!r} match no date format, e.g. {examples}")


def create_time_delta(df: pl.DataFrame, config: dict) -> pl.Expr:
    if "time" in config:
        if any(name in config for name in TIME_COMPONENTS):
            raise ValueError("Configure either time or hour/minute/second columns, not both")
        column = config["time"]
        if df.schema[column] == pl.Time:
            time = pl.col(column)
        else:
            values = pl.col(column).cast(pl.String).str.strip_chars()
            time = pl.coalesce(values.str.to_time(fmt, strict=False) for fmt in TIME_FORMATS)
        # Time casts to nanoseconds since midnight; a missing or invalid time counts as midnight.
        return pl.duration(nanoseconds=time.cast(pl.Int64).fill_null(0))

    microseconds = pl.lit(0, pl.Int64)
    resolved = pl.lit(True)
    for name, (unit, maximum) in TIME_COMPONENTS.items():
        if name not in config:
            break

        values = as_number(df, config[name])
        # Once a component is missing or out of range, finer components are ignored too.
        resolved = resolved & values.is_between(0, maximum)
        microseconds = microseconds + pl.when(resolved).then((values * unit).round().cast(pl.Int64)).otherwise(0)

    return pl.duration(microseconds=microseconds)
