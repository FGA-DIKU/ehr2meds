from typing import Dict, List, Tuple

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import CODE, MANDATORY_COLUMNS, SUBJECT_ID
from ehr2meds.PREMEDS.preprocessing.io.data_handling import (
    DataHandler,
    load_mapping_file,
)


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


def factorize_subject_id(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Factorize the subject_id column into an integer mapping."""
    df["integer_id"], unique_vals = pd.factorize(df[SUBJECT_ID])
    shifted_indices = df["integer_id"] + 1  # +1 to avoid binary dtype
    hash_to_int_map = dict(zip(unique_vals, shifted_indices))
    # Overwrite subject_id with the factorized integer and drop the helper column.
    df[SUBJECT_ID] = shifted_indices
    df = df.drop(columns=["integer_id"])
    return df, hash_to_int_map


def apply_secondary_mapping(
    df: pd.DataFrame, concept_config: dict, data_handler: "DataHandler"
) -> pd.DataFrame:
    """Apply secondary mapping (e.g., vnr to drug name)."""
    if "secondary_mapping" not in concept_config:
        return df

    mapping_cfg = concept_config["secondary_mapping"]
    mapping_df = load_mapping_file(mapping_cfg, data_handler)

    code_col = mapping_cfg.get("code_column")
    if code_col:
        df = apply_code_mapping(df, mapping_df, mapping_cfg, code_col)
    else:
        df = apply_simple_mapping(df, mapping_df, mapping_cfg)

    return df


def apply_code_mapping(
    df: pd.DataFrame, mapping_df: pd.DataFrame, mapping_cfg: dict, code_col: str
) -> pd.DataFrame:
    """Apply mapping with specific code column handling."""
    df = pd.merge(
        df,
        mapping_df[[mapping_cfg.get("right_on"), code_col]],
        left_on=mapping_cfg.get("left_on"),
        right_on=mapping_cfg.get("right_on"),
        how="inner",
    )
    df[CODE] = df[code_col]
    df = df.drop(columns=[mapping_cfg.get("left_on"), code_col])
    return df

def apply_mapping(df, map_table, join_col, source_col, target_col, rename_to=None, how="inner", drop_source=False):
    """
    Apply a mapping between two dataframes by joining them and optionally renaming/dropping columns.

    Args:
        df (pd.DataFrame): The main dataframe to apply the mapping to
        map_table (pd.DataFrame): The mapping table containing the values to map to
        join_col (str): The column in map_table to join on
        source_col (str): The column in df to join on
        target_col (str): The column from map_table to keep after joining
        rename_to (str, optional): New name for the target column after joining. Defaults to None.
        how (str, optional): Type of join to perform ('inner', 'left', etc). Defaults to "inner".
        drop_source (bool, optional): Whether to drop the source column after joining. Defaults to False.

    Returns:
        pd.DataFrame: The input dataframe with the mapping applied - joined with map_table
                     and cleaned up according to the parameters.

    Example:
        # Map patient IDs from one system to another
        df = apply_mapping(df, 
                         id_mapping_table,
                         join_col='old_id',
                         source_col='patient_id', 
                         target_col='new_id',
                         rename_to='patient_id',
                         drop_source=True)
    """
    # Perform the mapping
    df = pd.merge(
        df,
        map_table[[join_col, target_col]],  # Only select needed columns
        left_on=source_col,
        right_on=join_col,
        how=how
    )

    # Clean up intermediate columns
    if join_col != source_col:  # Avoid dropping if they're the same
        df = df.drop(columns=[join_col])

    # Optionally remove the original source column
    if drop_source:
        df = df.drop(columns=[source_col])
    
    # Rename the target column if requested
    if rename_to:
        df = df.rename(columns={target_col: rename_to})

    return df


def apply_simple_mapping(
    df: pd.DataFrame, mapping_df: pd.DataFrame, mapping_cfg: dict
) -> pd.DataFrame:
    """Apply simple mapping without code column."""
    df = pd.merge(
        df,
        mapping_df,
        left_on=mapping_cfg.get("left_on"),
        right_on=mapping_cfg.get("right_on"),
        how="inner",
    )
    df = df.drop(columns=[mapping_cfg.get("left_on")])
    return df


def convert_numeric_columns(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    """Convert specified columns to numeric type."""
    numeric_cols = concept_config.get("numeric_columns", [])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def map_and_clean_data(df: pd.DataFrame, subject_id_mapping: dict) -> pd.DataFrame:
    """Map from SP PIDs to integer subject_ids and clean the data."""
    # Map from SP PIDs to integer subject_ids
    if SUBJECT_ID in df.columns:
        df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
        # Drop rows where mapping failed (subject_id is NaN)
        df = df.dropna(subset=[SUBJECT_ID])
        # Convert subject_id to integer
        df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)

    # Clean data
    if all(col in df.columns for col in MANDATORY_COLUMNS):
        df = df.dropna(subset=MANDATORY_COLUMNS, how="any")

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def unroll_columns(df: pd.DataFrame, concept_config: dict) -> List[pd.DataFrame]:
    """
    Unroll specified columns into separate dataframes with code format.

    Returns a list of dataframes, each representing an unrolled column.
    """
    processed_dfs = []

    # Required columns to keep in each unrolled dataframe
    required_cols = [SUBJECT_ID]
    if "timestamp" in df.columns:
        required_cols.append("timestamp")

    # Keep only the columns that exist in the dataframe
    required_cols = [col for col in required_cols if col in df.columns]

    # For each column to unroll, create a separate df with it as CODE
    for col_info in concept_config["unroll_columns"]:
        col_name = col_info.get("column")
        if col_name in df.columns:
            # Create a copy with just the required columns and the unroll column
            unroll_df = df[required_cols + [col_name]].copy()

            # Apply prefix if specified
            prefix = col_info.get("prefix", "")

            # Rename to CODE
            unroll_df = unroll_df.rename(columns={col_name: CODE})

            # Add prefix to codes
            if prefix:
                unroll_df[CODE] = prefix + unroll_df[CODE].astype(str)

            processed_dfs.append(unroll_df)

    return processed_dfs
