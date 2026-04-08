import pandas as pd

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

    base = expanded_table[match_on + bound_cols].copy()
    base["__row_id"] = range(len(expanded_table))

    linked = df.copy()
    linked[timestamp] = pd.to_datetime(linked[timestamp], errors="coerce")

    merged = base.merge(linked, on=match_on, how="left")

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


def or_columns(expanded_table: pd.DataFrame, name: str, components: list[dict]) -> pd.DataFrame:
    """Boolean OR across component columns (NaN -> False)."""
    cols = components
    result = expanded_table.copy()
    result[name] = expanded_table[cols].fillna(False).astype(bool).any(axis=1).astype(bool)
    return result

def and_columns(expanded_table: pd.DataFrame, name: str, components: list[dict]) -> pd.DataFrame:
    """Boolean AND across component columns (NaN -> False)."""
    cols = components
    result = expanded_table.copy()
    result[name] = expanded_table[cols].fillna(False).astype(bool).all(axis=1).astype(bool)
    return result

def add_column(expanded_table: pd.DataFrame, name: str, components: list[dict]) -> pd.DataFrame:
    """Add component columns together."""
    cols = components
    result = expanded_table.copy()
    result[name] = expanded_table[cols]
    return result