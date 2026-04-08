import pandas as pd

def extract_GA(df, target_col):
    extracted = df[target_col].astype(str).str.extract(r"(?i)(\d+)\s*w", expand=False)
    df[target_col] = pd.to_numeric(extracted, errors="coerce")
    return df

def extract_regex_matches(df, target_col: str, regex: str):
    """Keep rows where ``target_col`` matches ``regex`` (substring); columns unchanged."""
    mask = df[target_col].astype(str).str.contains(regex, regex=True, na=False)
    return df.loc[mask].copy()

def extract_codes(df, target_col: str, match_on: list[str], match_type: str):
    """Filter rows by ``target_col``. ``match_on`` is a list; a row matches if **any** option matches."""
    s = df[target_col].astype(str)
    if match_type == "exact":
        mask = s.isin(match_on)
    elif match_type == "startswith":
        mask = pd.Series(False, index=s.index)
        for p in match_on:
            mask |= s.str.startswith(p, na=False)
    elif match_type == "endswith":
        mask = pd.Series(False, index=s.index)
        for p in match_on:
            mask |= s.str.endswith(p, na=False)
    elif match_type == "contains":
        mask = pd.Series(False, index=s.index)
        for p in match_on:
            mask |= s.str.contains(p, regex=False, na=False)
    else:
        raise ValueError(f"Invalid match type: {match_type}")
    return df.loc[mask].copy()

def fill_matches(
    df,
    target_col: str,
    match_on: list[str],
    match_type: str,
    fill_value,
    fill_col: str = "fill_col",
):
    """Same row filter as ``extract_codes``; adds ``fill_col`` = ``fill_value`` on each kept row."""
    out = extract_codes(df, target_col, match_on, match_type)
    out[fill_col] = fill_value
    return out

def extract_non_nan(df, target_col: str, fill_value, fill_col: str = "fill_col"):
    """Keep rows where ``target_col`` is non-null; add ``fill_col`` = ``fill_value`` on each row."""
    mask = df[target_col].notna()
    out = df.loc[mask].copy()
    out[fill_col] = fill_value
    return out

def fill_bool_match(
    df: pd.DataFrame,
    target_col: str,
    op: str,
    val,
    fill_value,
    fill_col: str = "fill_col",
) -> pd.DataFrame:
    """
    Keep rows where a simple boolean condition on ``target_col`` is true; add ``fill_col``.

    - ``op``: one of "==", "!=", ">=", "<=", ">", "<"
    - ``val``: numeric threshold (int/float). ``target_col`` is coerced to numeric with errors -> NaN
      (which never matches).
    """
    try:
        val_num = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"val must be numeric; got {val!r}") from e

    s = pd.to_numeric(df[target_col], errors="coerce")
    if op == "==":
        mask = s == val_num
    elif op == "!=":
        mask = s != val_num
    elif op == ">=":
        mask = s >= val_num
    elif op == "<=":
        mask = s <= val_num
    elif op == ">":
        mask = s > val_num
    elif op == "<":
        mask = s < val_num
    else:
        raise ValueError(f"Unsupported operator {op!r}; expected one of ==, !=, >=, <=, >, <")

    out = df.loc[mask].copy()
    out[fill_col] = fill_value
    return out

def get_pregnancy_start(birthdate, GA):
    """Pregnancy start = delivery date minus gestational age (works on Series or scalars)."""
    ga_weeks = pd.to_numeric(GA, errors="coerce")
    return pd.to_datetime(birthdate) - pd.to_timedelta(ga_weeks * 7, unit="d")

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

def extract_columns(df, target_cols: list[str]):
    """Extract columns from dataframe."""
    return df[target_cols]

def get_time_difference(df, start_time: str, end_time: str, unit: str):
    """Compute time difference between start and end time."""
    df[start_time] = pd.to_datetime(df[start_time], errors="coerce")
    df[end_time] = pd.to_datetime(df[end_time], errors="coerce")
    return (df[end_time] - df[start_time]).dt.total_seconds() / (3600 * 24 * getattr(pd.Timedelta, unit))

def latest_entry(df, target_col: str, date_col: str, max_date: str):
    """Get latest entry for each group defined by ``required_cols``."""
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col] <= pd.to_datetime(max_date, errors="coerce")]
    return df[target_col].max()