

import pandas as pd


def merge_on_match_on(
    df: pd.DataFrame,
    expanded_table: pd.DataFrame,
    match_on: list[str],
    extra_cols: list[str] | None = None,
):
    """Merge df onto expanded_table using `match_on` keys, carrying `extra_cols` from expanded_table."""
    cols = list(match_on)
    if extra_cols:
        cols.extend([c for c in extra_cols if c in expanded_table.columns])
    # Always require match keys to exist on expanded_table.
    base = expanded_table[cols].copy()
    base["__row_id"] = range(len(expanded_table))
    merged = base.merge(df, on=match_on, how="left")
    return merged


def _merge_rows_matching_bool_in_time_window(
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    timestamp: str,
    max_date: str | None = None,
    min_date: str | None = None,
) -> pd.DataFrame:
    """
    Merge linked ``df`` onto ``expanded_table`` and keep rows that pass the same
    time-window rules as :func:`bool_in_time_window` (including non-null ``timestamp``).
    """
    bound_cols: list[str] = []
    if min_date is not None:
        bound_cols.append(min_date)
    if max_date is not None:
        bound_cols.append(max_date)

    merged = merge_on_match_on(df, expanded_table, match_on=match_on, extra_cols=bound_cols)

    merged[timestamp] = pd.to_datetime(merged[timestamp], errors="coerce")

    if min_date is not None:
        merged[min_date] = pd.to_datetime(merged[min_date], errors="coerce")
        merged = merged[merged[timestamp] >= merged[min_date]]

    if max_date is not None:
        merged[max_date] = pd.to_datetime(merged[max_date], errors="coerce")
        merged = merged[merged[timestamp] <= merged[max_date]]

    merged = merged.loc[merged[timestamp].notna()]
    return merged


def bool_in_time_window(
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    timestamp: str,
    max_date: str | None = None,
    min_date: str | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """
    For each row in expanded_table, return True if df contains at least one matching row
    (based on match_on keys) whose `timestamp` falls within the per-row [min_date, max_date]
    window defined by columns in expanded_table. Bounds are optional.

    - match_on: list of key columns present in both df and expanded_table
    - timestamp: column in df containing the event date/time
    - min_date/max_date: column names in expanded_table (not literals)
    """
    merged = _merge_rows_matching_bool_in_time_window(
        df, match_on, expanded_table, timestamp, max_date=max_date, min_date=min_date
    )

    # True if any linked rows remain for that expanded row id.
    kept_row_ids = merged["__row_id"].drop_duplicates()
    out = pd.Series(False, index=expanded_table.index, dtype=bool)
    out.iloc[kept_row_ids.to_numpy()] = True

    col_name = name or "name"
    result = expanded_table.copy()
    result[col_name] = out
    return result

def bool_in_time_shifted_window(
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    timestamp: str,
    max_date: str | None = None,
    min_date: str | None = None,
    shift_max_days: int | None = None,
    shift_min_days: int | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """
    Shift optional lower/upper bounds by calendar days, then delegate to ``bool_in_time_window``.

    Apply shifts on one ``expanded_table`` copy so min and max shifts compose correctly.
    """
    if shift_min_days is None and shift_max_days is None:
        return bool_in_time_window(
            df=df,
            match_on=match_on,
            expanded_table=expanded_table,
            timestamp=timestamp,
            max_date=max_date,
            min_date=min_date,
            name=name,
        )

    shifted = expanded_table.copy()
    out_min = min_date
    out_max = max_date

    if shift_min_days is not None:
        shifted_min_col = f"__shifted_{min_date}"
        shifted[shifted_min_col] = pd.to_datetime(shifted[min_date], errors="coerce") - pd.to_timedelta(
            shift_min_days, unit="d"
        )
        out_min = shifted_min_col

    if shift_max_days is not None:
        shifted_max_col = f"__shifted_{max_date}"
        shifted = expanded_table.copy()
        shifted[shifted_max_col] = pd.to_datetime(shifted[max_date], errors="coerce") + pd.to_timedelta(
            shift_max_days, unit="d"
        )
        out_max = shifted_max_col

    return bool_in_time_window(
        df=df,
        match_on=match_on,
        expanded_table=shifted,
        timestamp=timestamp,
        max_date=out_max,
        min_date=out_min,
        name=name,
    )

def extract_column(    
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    target_col: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Extract column from dataframe."""
    merged = merge_on_match_on(df, expanded_table, match_on=match_on)
    expanded_table[name] = merged[target_col]
    return expanded_table

# Seconds per calendar unit (approximate; matches prior get_time_difference behaviour).
_SECONDS_PER_YEAR = 3600 * 24 * 365.25
_SECONDS_PER_MONTH = 3600 * 24 * (365.25 / 12.0)
# Divisors for total_seconds() to express a timedelta in the given unit.
_UNIT_SECONDS = {
    "days": 86400.0,
    "hours": 3600.0,
    "minutes": 60.0,
    "seconds": 1.0,
    "weeks": 86400.0 * 7,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}


def get_time_difference(
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    start_time: str,
    end_time: str,
    unit: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Compute time difference between start and end time."""
    col_name = name or "name"
    merged = merge_on_match_on(
        df, expanded_table, match_on=match_on, extra_cols=[start_time, end_time]
    )
    missing = [c for c in (start_time, end_time) if c not in merged.columns]
    if missing:
        raise KeyError(
            "get_time_difference: missing column(s) after merge: "
            f"{missing}; start_time={start_time!r}, end_time={end_time!r}; "
            f"have {list(merged.columns)}"
        )

    merged[start_time] = pd.to_datetime(merged[start_time], errors="coerce")
    merged[end_time] = pd.to_datetime(merged[end_time], errors="coerce")
    try:
        delta = merged[end_time] - merged[start_time]
    except Exception as e:
        raise ValueError(
            f"get_time_difference: could not subtract {end_time!r} - {start_time!r}: {e}"
        ) from e

    if not hasattr(delta, "dt"):
        raise TypeError(
            f"get_time_difference: expected a timedelta Series for "
            f"{end_time!r} - {start_time!r}, got {type(delta).__name__}"
        )

    unit_norm = str(unit).lower().strip()
    try:
        if unit_norm == "years":
            merged[col_name] = delta.dt.total_seconds() / _SECONDS_PER_YEAR
        elif unit_norm == "months":
            merged[col_name] = delta.dt.total_seconds() / _SECONDS_PER_MONTH
        elif unit_norm in _UNIT_SECONDS:
            merged[col_name] = delta.dt.total_seconds() / _UNIT_SECONDS[unit_norm]
        else:
            raise ValueError(
                "get_time_difference: unsupported unit "
                f"{unit!r} (normalized {unit_norm!r}). "
                f"Use 'years', 'months', or one of: {sorted(_UNIT_SECONDS)}"
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"get_time_difference: failed to convert delta to unit {unit!r}: {e}"
        ) from e

    return merged

def get_GA_at_date(
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    date: str,
    GA_col: str,
    pregnancy_start: str,
    pregnancy_end: str,
    name: str | None = None,
) -> pd.DataFrame:
    """
    Estimate GA in weeks at ``date`` using GA at delivery (``GA_col``, weeks) and
    ``pregnancy_end`` as the delivery time. Only merged rows with
    ``pregnancy_start <= date <= pregnancy_end`` are kept. If several linked rows
    qualify for one expanded row, the row with the latest ``date`` is used.
    """
    if date not in df.columns or GA_col not in df.columns:
        raise KeyError(
            f"expected {date!r} and {GA_col!r} on linked df; have {list(df.columns)}"
        )

    merged = merge_on_match_on(
        df, expanded_table, match_on=match_on, extra_cols=[pregnancy_start, pregnancy_end]
    )
    merged[date] = pd.to_datetime(merged[date], errors="coerce")
    merged[pregnancy_start] = pd.to_datetime(merged[pregnancy_start], errors="coerce")
    merged[pregnancy_end] = pd.to_datetime(merged[pregnancy_end], errors="coerce")

    in_window = (
        merged[date].notna()
        & merged[pregnancy_start].notna()
        & merged[pregnancy_end].notna()
        & (merged[date] >= merged[pregnancy_start])
        & (merged[date] <= merged[pregnancy_end])
    )
    merged = merged.loc[in_window].copy()
    ga_at_delivery = pd.to_numeric(merged[GA_col], errors="coerce")
    weeks_to_end = (merged[pregnancy_end] - merged[date]).dt.days / 7.0
    merged["_GA_at_date"] = ga_at_delivery - weeks_to_end

    col_name = name or "name"
    result = expanded_table.copy()
    out = pd.Series([pd.NA] * len(expanded_table), dtype=object, index=expanded_table.index)

    if merged.empty:
        result[col_name] = out
        return result

    idx = merged.groupby("__row_id", sort=False)[date].idxmax()
    picked = merged.loc[idx]
    row_ids = picked["__row_id"].to_numpy()
    vals = picked["_GA_at_date"].to_numpy()
    out.iloc[row_ids] = vals
    result[col_name] = out
    return result

def latest_entry(df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    target_col: str,
    date_col: str,
    max_date: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Get latest entry for each group defined by ``required_cols``."""
    merged = merge_on_match_on(df, expanded_table, match_on=match_on, extra_cols=[date_col, max_date])
    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
    merged = merged[merged[date_col] <= pd.to_datetime(max_date, errors="coerce")]
    expanded_table[name] = merged[target_col].max()
    return expanded_table

def is_present_bool(
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    target_col: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Return True if target_col is present in df."""
    merged = merge_on_match_on(df, expanded_table, match_on=match_on)
    expanded_table[name] = merged[target_col].notna().astype(bool)
    return expanded_table

def value_in_time_window(
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    timestamp: str,
    target_col: str,
    max_date: str | None = None,
    min_date: str | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """
    Reuse the same linked rows as ``bool_in_time_window``; for each expanded row where
    that function would be True, set ``name`` to ``target_col`` from the **latest**
    matching linked row (by ``timestamp``). Otherwise missing.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"target_col {target_col!r} not in linked dataframe; have {list(df.columns)}"
        )

    merged = _merge_rows_matching_bool_in_time_window(
        df, match_on, expanded_table, timestamp, max_date=max_date, min_date=min_date
    )
    col_name = name or "name"
    result = expanded_table.copy()
    out = pd.Series([pd.NA] * len(expanded_table), dtype=object, index=expanded_table.index)

    if merged.empty:
        result[col_name] = out
        return result

    idx = merged.groupby("__row_id", sort=False)[timestamp].idxmax()
    picked = merged.loc[idx]
    row_ids = picked["__row_id"].to_numpy()
    vals = picked[target_col].to_numpy()
    out.iloc[row_ids] = vals
    result[col_name] = out
    return result