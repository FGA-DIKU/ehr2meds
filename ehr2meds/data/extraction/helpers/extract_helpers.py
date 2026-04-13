import pandas as pd

import ehr2meds.data.extraction.helpers.filter_helpers as filter_helpers

def extract_GA(df, target_col):
    extracted = df[target_col].astype(str).str.extract(r"(?i)(\d+)\s*w", expand=False)
    df[target_col] = pd.to_numeric(extracted, errors="coerce")
    return df

def extract_regex_matches(df, target_col: str, regex: str):
    """Keep rows where ``target_col`` matches ``regex`` (substring); columns unchanged."""
    mask = df[target_col].astype(str).str.contains(regex, regex=True, na=False)
    return df.loc[mask].copy()

def extract_codes(df, target_col: str, match_on: list[str], match_type: str, exclude:bool=False):
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
    if exclude:
        mask = ~mask
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

def get_pregnancy_start(birthdate, GA):
    """Pregnancy start = delivery date minus gestational age (works on Series or scalars)."""
    ga_weeks = pd.to_numeric(GA, errors="coerce")
    return pd.to_datetime(birthdate) - pd.to_timedelta(ga_weeks * 7, unit="d")