import pandas as pd

def bool_in_time_window(
    df: pd.DataFrame,
    match_on: list[str],
    expanded_table: pd.DataFrame,
    timestamp: str,
    max_date: str | None = None,
    min_date: str | None = None,
) -> pd.Series:
    """
    For each row in expanded_table, return True if df contains at least one matching row
    (based on match_on keys) whose `timestamp` falls within the per-row [min_date, max_date]
    window defined by columns in expanded_table. Bounds are optional.

    - match_on: list of key columns present in both df and expanded_table
    - timestamp: column in df containing the event date/time
    - min_date/max_date: column names in expanded_table (not literals)
    """
    if not match_on:
        raise ValueError("match_on must be a non-empty list of key columns.")
    missing_main = [c for c in match_on if c not in expanded_table.columns]
    if missing_main:
        raise ValueError(
            f"expanded_table missing match_on columns {missing_main}; "
            f"available: {list(expanded_table.columns)}"
        )
    missing_linked = [c for c in match_on if c not in df.columns]
    if missing_linked:
        raise ValueError(
            f"df missing match_on columns {missing_linked}; "
            f"available: {list(df.columns)}"
        )
    if timestamp not in df.columns:
        raise ValueError(
            f"df missing timestamp column {timestamp!r}; available: {list(df.columns)}"
        )
    if max_date is not None and max_date not in expanded_table.columns:
        raise ValueError(
            f"expanded_table missing max_date column {max_date!r}; "
            f"available: {list(expanded_table.columns)}"
        )
    if min_date is not None and min_date not in expanded_table.columns:
        raise ValueError(
            f"expanded_table missing min_date column {min_date!r}; "
            f"available: {list(expanded_table.columns)}"
        )

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
    return out