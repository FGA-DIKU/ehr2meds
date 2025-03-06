import pandas as pd

from MEDS_preprocess.preprocessing.constants import CODE


def select_and_rename_columns(df: pd.DataFrame, columns_map: dict) -> pd.DataFrame:
    """Select and rename columns based on columns_map."""
    check_columns(df, columns_map)
    df = df[list(columns_map.keys())]
    df = df.rename(columns=columns_map)
    return df


def process_codes(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    """Filling missing values, and adding prefixes."""
    # Fill missing values
    fillna_cfg = concept_config.get("fillna")
    if fillna_cfg:
        df = fill_missing_values(df, fillna_cfg)

    # Add code prefix if configured
    code_prefix = concept_config.get("code_prefix", "")
    if code_prefix and CODE in df.columns:
        df[CODE] = code_prefix + df[CODE].astype(str)

    return df


def fill_missing_values(df: pd.DataFrame, fillna_cfg: dict) -> pd.DataFrame:
    """
    Fill missing values using specified columns and regex patterns.
    Drop the columns used to fill missing values.
    """
    for target_col, fill_config in fillna_cfg.items():
        fill_col = fill_config.get("column")
        if fill_col and fill_col in df.columns:
            fillna_regex = fill_config.get("regex")
            if fillna_regex:
                fill_vals = df[fill_col].str.extract(fillna_regex, expand=False)
            else:
                fill_vals = df[fill_col]
            df[target_col] = df[target_col].fillna(fill_vals)
            df = df.drop(columns=[fill_col])
    return df


def check_columns(df: pd.DataFrame, columns_map: dict):
    """Check if all columns in columns_map are present in df."""
    missing_columns = set(columns_map.keys()) - set(df.columns)
    if missing_columns:
        available_columns = pd.DataFrame({"Available Columns": sorted(df.columns)})
        requested_columns = pd.DataFrame(
            {"Requested Columns": sorted(columns_map.keys())}
        )
        error_msg = f"\nMissing columns: {sorted(missing_columns)}\n\n"
        error_msg += "Columns comparison:\n"
        error_msg += (
            f"{pd.concat([available_columns, requested_columns], axis=1).to_string()}"
        )
        raise ValueError(error_msg)
