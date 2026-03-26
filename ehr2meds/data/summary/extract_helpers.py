


import numpy as np
import pandas as pd


def latest_entry(df, name, required_cols, target_col, date_col):
    """
    For each group defined by ``required_cols``, return the ``target_col`` value
    from the row with the latest timestamp in ``date_col``.

    The output column is named ``name``.
    """
    group_cols = list(required_cols)
    cols = group_cols + [target_col, date_col]
    d = df[cols].copy()

    # Ensure the date column can be compared.
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    # Sort ascending, with NaT first, then take the last row per group.
    d = d.sort_values(date_col, ascending=True, na_position="first")
    latest_rows = d.groupby(group_cols, as_index=False).tail(1)

    out = latest_rows[group_cols + [target_col]].rename(columns={target_col: name})
    return out

def has_one_of_values(df, name, required_cols, target_col):
    group_cols = list(required_cols)
    return df[group_cols].drop_duplicates().assign(**{name: 1})

def has_all_values(df, name, required_cols, target_col, include_list):
    """
    For each group in ``required_cols``, set ``name`` to 1 if every value in
    ``include_list`` appears at least once in ``target_col`` within that group;
    otherwise ``name`` is NaN (not 0).
    """
    group_cols = list(required_cols)
    wanted = set(include_list)
    all_groups = df[group_cols].drop_duplicates()

    present = (
        df[df[target_col].isin(wanted)]
        .groupby(group_cols)[target_col]
        .nunique()
        .reset_index(name="_n_present")
    )
    out = all_groups.merge(present, on=group_cols, how="left")
    n_present = out["_n_present"].fillna(0).astype(int)
    out[name] = np.where(n_present == len(wanted), 1, np.nan)
    return out.drop(columns=["_n_present"])

# def extract_latest_value(df, name, required_cols, target_col, match_on, extract_col, date_col):
#     group_cols = list(required_cols)
#     cols = group_cols + [target_col, match_on, extract_col, date_col]
#     d = df[cols].copy()

#     d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
#     d = d.sort_values(date_col, ascending=True, na_position="first")
#     latest_rows = d.groupby(group_cols, as_index=False).tail(1)
#     out = latest_rows[group_cols + [extract_col]].rename(columns={extract_col: name})
#     return out