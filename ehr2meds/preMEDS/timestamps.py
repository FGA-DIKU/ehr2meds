"""Construction of complete timestamps from separate date and time columns."""

import pandas as pd


def add_timestamp_columns(df: pd.DataFrame, timestamp_config: dict) -> pd.DataFrame:
    """Combine dates with valid times; otherwise retain the date at midnight."""
    for output_column, config in timestamp_config.items():
        date = pd.to_datetime(df[config["date"]], errors="coerce").dt.normalize()
        time, valid_time = create_time_delta(df, config)
        timestamp = date + time.where(valid_time, pd.Timedelta(0))
        df[output_column] = timestamp.astype("datetime64[us]")
    return df


def create_time_delta(df: pd.DataFrame, config: dict) -> tuple[pd.Series, pd.Series]:
    component_names = [name for name in ("hour", "minute", "second") if name in config]
    if "time" in config:
        if component_names:
            raise ValueError("Configure either time or hour/minute/second columns, not both")
        values = df[config["time"]]
        delta = pd.to_timedelta(values.astype("string").str.strip(), errors="coerce")
        return delta, values.notna() & delta.notna()

    if not component_names:
        zero = pd.Series(pd.Timedelta(0), index=df.index)
        return zero, pd.Series(False, index=df.index)

    components = {name: pd.to_numeric(df[config[name]], errors="coerce") for name in component_names}
    valid_time = pd.Series(True, index=df.index)
    for values in components.values():
        valid_time = valid_time & values.notna()

    if "hour" in components:
        valid_time = valid_time & components["hour"].between(0, 23)
    if "minute" in components:
        valid_time = valid_time & components["minute"].between(0, 59)
    if "second" in components:
        valid_time = valid_time & components["second"].between(0, 59)

    time = pd.Series(pd.Timedelta(0), index=df.index)
    for name, values in components.items():
        unit = {"hour": "h", "minute": "m", "second": "s"}[name]
        time = time + pd.to_timedelta(values.fillna(0), unit=unit)

    return time, valid_time
