"""Construction of complete timestamps from separate date and time columns."""

import pandas as pd

# Pandas timedelta unit and largest valid value, ordered from coarsest to finest.
TIME_COMPONENTS = {"hour": ("h", 23), "minute": ("m", 59), "second": ("s", 59)}

def add_timestamp_columns(df: pd.DataFrame, timestamp_config: dict) -> pd.DataFrame:
    """Combine each date with as much valid time information as is available."""
    for output_column, config in timestamp_config.items():
        date = pd.to_datetime(
            df[config["date"]],
            format=config.get("format"),
            errors="coerce",
        ).dt.normalize()

        timestamp = date + create_time_delta(df, config)
        df[output_column] = timestamp.astype("datetime64[us]")

    return df



def create_time_delta(df: pd.DataFrame, config: dict) -> pd.Series:
    if "time" in config:
        if any(name in config for name in TIME_COMPONENTS):
            raise ValueError("Configure either time or hour/minute/second columns, not both")
        values = df[config["time"]]
        time = pd.to_timedelta(values.astype("string").str.strip(), errors="coerce")
        return time.fillna(pd.Timedelta(0))

    time = pd.Series(pd.Timedelta(0), index=df.index)
    resolved = pd.Series(True, index=df.index)
    for name, (unit, maximum) in TIME_COMPONENTS.items():
        if name not in config:
            break

        values = pd.to_numeric(df[config[name]], errors="coerce")
        resolved = resolved & values.between(0, maximum)
        time = time + pd.to_timedelta(values.where(resolved, 0), unit=unit)

    return time