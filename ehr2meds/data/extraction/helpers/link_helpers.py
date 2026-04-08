

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
    # Keep a stable row id to collapse back to one boolean per expanded row after merge
    # (merge can create multiple rows per expanded_table row).
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

    # True if any linked rows remain for that expanded row id.
    kept_row_ids = merged.loc[merged[timestamp].notna(), "__row_id"].drop_duplicates()
    out = pd.Series(False, index=expanded_table.index, dtype=bool)
    out.iloc[kept_row_ids.to_numpy()] = True

    col_name = name or "name"
    result = expanded_table.copy()
    result[col_name] = out
    return result

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

def get_time_difference(df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    start_time: str,
    end_time: str,
    unit: str,
    name: str | None = None,
) -> pd.DataFrame:
    """Compute time difference between start and end time."""
    merged = merge_on_match_on(df, expanded_table, match_on=match_on, extra_cols=[start_time, end_time])
    merged[start_time] = pd.to_datetime(merged[start_time], errors="coerce")
    merged[end_time] = pd.to_datetime(merged[end_time], errors="coerce")
    delta = merged[end_time] - merged[start_time]
    unit_norm = str(unit).lower()
    if unit_norm in {"years"}: #pandas Timedelta not supporting "years".
        merged[name] = delta.dt.total_seconds() / (3600 * 24 * 365.25)
    elif unit_norm in {"months"}:
        merged[name] = delta.dt.total_seconds() / (3600 * 24 * (365.25 / 12.0))
    else:
        merged[name] = delta.dt.total_seconds() / (3600 * 24 * getattr(pd.Timedelta, unit_norm))
    return merged

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
    merged[name] = merged[target_col].max()
    return merged