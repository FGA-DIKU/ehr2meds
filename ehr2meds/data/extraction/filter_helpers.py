

import pandas as pd


def drop_empty_columns(df, columns: list[str]):
    """Drop columns that are empty."""
    return df.dropna(subset=columns)

def merge_on_match_on(df: pd.DataFrame, expanded_table: pd.DataFrame, match_on: list[str]):
    """Merge df onto expanded_table on match_on keys."""
    base = expanded_table[match_on].copy()
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

    merged = merge_on_match_on(df, expanded_table, match_on + bound_cols)

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

def extract_columns(    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    target_cols: list[str],
    name: str | None = None,
) -> pd.DataFrame:
    """Extract columns from dataframe."""
    merged = merge_on_match_on(df, expanded_table, match_on)
    merged[name] = merged[target_cols]
    return merged

# def get_time_difference(df, start_time: str, end_time: str, unit: str):
#     """Compute time difference between start and end time."""
#     df[start_time] = pd.to_datetime(df[start_time], errors="coerce")
#     df[end_time] = pd.to_datetime(df[end_time], errors="coerce")
#     return (df[end_time] - df[start_time]).dt.total_seconds() / (3600 * 24 * getattr(pd.Timedelta, unit))

# def latest_entry(df, target_col: str, date_col: str, max_date: str):
#     """Get latest entry for each group defined by ``required_cols``."""
#     df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
#     df = df[df[date_col] <= pd.to_datetime(max_date, errors="coerce")]
#     return df[target_col].max()