import pandas as pd


def extract_column(df, target_col):
    return df[target_col]


def extract_GA(df, target_col):
    """Parse leading week count from strings like '17w0d' or '9w0d' into integers."""
    extracted = df[target_col].astype(str).str.extract(r"(?i)(\d+)\s*w", expand=False)
    df[target_col] = pd.to_numeric(extracted, errors="coerce")
    return df