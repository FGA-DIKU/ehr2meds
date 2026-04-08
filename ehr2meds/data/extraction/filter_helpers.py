

import pandas as pd


def _match_mask(s, values, match):
    """Boolean mask: True where ``s`` matches any entry in ``values``."""
    if match == "exact_match":
        return s.isin(values)
    if match == "startswith":
        return s.astype(str).str.startswith(tuple(values), na=False)
    if match == "contain":
        ser = s.astype(str)
        m = pd.Series(False, index=s.index)
        for pat in values:
            m |= ser.str.contains(pat, regex=False, na=False)
        return m
    raise KeyError(match)


def include_exclude(df, target_col, include=None, exclude=None, match="exact_match"):
    """
    Filter rows by ``target_col`` using ``include`` / ``exclude`` lists.

    ``match`` must be exactly one of:
    ``exact_match``, ``startswith``, ``contain``.
    """
    out = df

    if include:
        s = out[target_col]
        out = out[_match_mask(s, list(include), match)]

    if exclude:
        s = out[target_col]
        out = out[~_match_mask(s, list(exclude), match)]

    return out

def drop_empty_columns(df, columns: list[str]):
    """Drop columns that are empty."""
    return df.dropna(subset=columns)
