import pandas as pd


def extract_columns(df, target_cols):
    """Ensure listed columns exist; full frame is passed through for ``map_columns`` selection."""
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}; available: {list(df.columns)}"
        )
    return df


def extract_GA(df, target_col):
    """Parse leading week count from strings like '17w0d' or '9w0d' into integers."""
    extracted = df[target_col].astype(str).str.extract(r"(?i)(\d+)\s*w", expand=False)
    df[target_col] = pd.to_numeric(extracted, errors="coerce")
    return df