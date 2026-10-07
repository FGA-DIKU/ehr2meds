"""Construction of complete timestamps from separate date and time columns."""

import pandas as pd


# Pandas timedelta unit and largest valid value, ordered from coarsest to finest.
TIME_COMPONENTS = {
    "hour": ("h", 23),
    "minute": ("m", 59),
    "second": ("s", 59),
}

# Common date formats consisting of exactly eight digits.
COMPACT_DATE_FORMATS = (
    "%Y%m%d",  # YYYYMMDD
    "%Y%d%m",  # YYYYDDMM
    "%d%m%Y",  # DDMMYYYY
    "%m%d%Y",  # MMDDYYYY
)


def parse_date_column(values: pd.Series) -> pd.Series:
    """Parse a date column, inferring compact 8-digit formats when unambiguous."""
    strings = values.astype("string").str.strip()
    non_null = strings.dropna()

    # Nothing to infer if the column contains no values.
    if non_null.empty:
        return pd.to_datetime(values, errors="coerce")

    # Special handling for compact dates such as 31122020.
    if non_null.str.fullmatch(r"\d{8}").all():
        # Infer the format from a sample rather than repeatedly parsing
        # potentially hundreds of millions of values.
        sample = non_null.sample(
            n=min(100_000, len(non_null)),
            random_state=0,
        )

        scores = {}

        for fmt in COMPACT_DATE_FORMATS:
            parsed = pd.to_datetime(
                sample,
                format=fmt,
                errors="coerce",
            )
            scores[fmt] = parsed.notna().sum()

        best_score = max(scores.values())

        best_formats = [
            fmt
            for fmt, score in scores.items()
            if score == best_score
        ]

        # Only infer the format if exactly one format successfully
        # parses every value in the sample.
        if len(best_formats) == 1 and best_score == len(sample):
            return pd.to_datetime(
                strings,
                format=best_formats[0],
                errors="coerce",
            )

    # Preserve the original behaviour for all other date columns.
    return pd.to_datetime(
        values,
        errors="coerce",
    )


def add_timestamp_columns(
    df: pd.DataFrame,
    timestamp_config: dict,
) -> pd.DataFrame:
    """Combine each date with as much valid time information as is available."""
    for output_column, config in timestamp_config.items():
        date = parse_date_column(
            df[config["date"]]
        ).dt.normalize()

        timestamp = date + create_time_delta(df, config)

        df[output_column] = timestamp.astype("datetime64[us]")

    return df


def create_time_delta(
    df: pd.DataFrame,
    config: dict,
) -> pd.Series:
    if "time" in config:
        if any(name in config for name in TIME_COMPONENTS):
            raise ValueError(
                "Configure either time or hour/minute/second columns, not both"
            )

        values = df[config["time"]]

        time = pd.to_timedelta(
            values.astype("string").str.strip(),
            errors="coerce",
        )

        return time.fillna(pd.Timedelta(0))

    time = pd.Series(
        pd.Timedelta(0),
        index=df.index,
    )

    resolved = pd.Series(
        True,
        index=df.index,
    )

    for name, (unit, maximum) in TIME_COMPONENTS.items():
        if name not in config:
            break

        values = pd.to_numeric(
            df[config[name]],
            errors="coerce",
        )

        resolved = resolved & values.between(
            0,
            maximum,
        )

        time = time + pd.to_timedelta(
            values.where(resolved, 0),
            unit=unit,
        )

    return time