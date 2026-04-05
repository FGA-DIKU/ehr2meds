import pandas as pd


def extract_columns(df, target_cols):
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}; available: {list(df.columns)}"
        )
    return df


def extract_GA(df, target_col):
    extracted = df[target_col].astype(str).str.extract(r"(?i)(\d+)\s*w", expand=False)
    df[target_col] = pd.to_numeric(extracted, errors="coerce")
    return df

def extract_codes(df, target_col, match_on):
    mask = df[target_col].astype(str).isin(match_on)
    return df.loc[mask].copy()