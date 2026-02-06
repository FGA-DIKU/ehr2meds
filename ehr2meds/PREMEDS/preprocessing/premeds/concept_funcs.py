import logging
import re
from typing import Dict, List, Tuple

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import (
    CODE,
    MANDATORY_COLUMNS,
    SUBJECT_ID,
    TIMESTAMP,
)

logger = logging.getLogger(__name__)


def select_and_rename_columns(df: pd.DataFrame, columns_map: dict) -> pd.DataFrame:
    """Select and rename columns based on columns_map."""
    check_columns(df, columns_map)
    df = df[list(columns_map.keys())]
    df = df.rename(columns=columns_map)
    return df


def prefix_codes(df: pd.DataFrame, code_prefix: str = None) -> pd.DataFrame:
    """Add a prefix to the entries in the code column."""
    if code_prefix and CODE in df.columns:
        df[CODE] = code_prefix + df[CODE].astype(str)
    return df


def fill_missing_values(df: pd.DataFrame, fillna_cfg: dict) -> pd.DataFrame:
    """
    Fill missing values using specified columns and regex patterns.
    Drop the columns used to fill missing values.
    
    Args:
        df: DataFrame to process
        fillna_cfg: Dictionary with target columns as keys and fill config as values
        
    Returns:
        DataFrame with missing values filled and fill columns dropped
    """
    if not fillna_cfg:
        return df
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
    """Factorize the SUBJECT_ID column into an integer mapping."""
    col = SUBJECT_ID
    
    # Validate all SUBJECT_ID values: check for non-scalar values and array-like strings
    bad_rows = [
        (idx, val, type(val).__name__, "non-scalar" if not pd.api.types.is_scalar(val) else "array-like string")
        for idx, val in df[col].items()
        if not pd.api.types.is_scalar(val) or (isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"))
    ]
    
    if bad_rows:
        raise ValueError(
            f"Invalid SUBJECT_ID values detected: {len(bad_rows)} row(s). "
            f"First problematic row: {bad_rows[0][0]}, value: {repr(bad_rows[0][1])}, type: {bad_rows[0][2]}"
        )
    
    # Create mapping from unique values to integers
    unique_vals = df[col].unique()
    hash_to_int_map = {val: idx + 2 for idx, val in enumerate(sorted(unique_vals))}  # +2 prevents binary interpretation
    
    # Apply mapping and validate
    mapped = df[col].map(hash_to_int_map)
    if mapped.isna().any():
        bad_idx = mapped[mapped.isna()].index.tolist()
        raise ValueError(
            f"Unmapped SUBJECT_ID values detected: {len(bad_idx)} row(s). "
            f"First unmapped row: {bad_idx[0]}, value: {repr(df.loc[bad_idx[0], col])}"
        )
    
    df[col] = mapped.astype(int)
    return df, hash_to_int_map




def apply_mapping(
    df,
    map_table,
    join_col,
    source_col,
    target_col,
    rename_to=None,
    how="inner",
    drop_source=False,
):
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
    # Ensure that join key columns are of the same type
    if df[source_col].dtype != map_table[join_col].dtype:
        df[source_col] = df[source_col].astype(str)
        map_table[join_col] = map_table[join_col].astype(str)

    # Perform the mapping
    df = pd.merge(
        df,
        map_table[[join_col, target_col]],  # Only select needed columns
        left_on=source_col,
        right_on=join_col,
        how=how,
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


def convert_numeric_columns(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    """Convert specified columns to numeric type."""
    numeric_cols = concept_config.get("numeric_columns", [])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def convert_datetime_columns(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    """Convert specified columns to datetime type. Handles both date-only and datetime formats.
    
    Date-only values like "2024-01-15" are automatically converted to "2024-01-15 00:00:00".
    If time_format is specified, it's used as a hint but values that don't match will be
    parsed without format to avoid data loss.
    
    Uses only the global datetime config from '_global_datetime' key with:
    - 'columns': list of column names
    - 'timeformat': time format string
    """
    # Get global datetime config only
    global_datetime = concept_config.get("_global_datetime", {})
    datetime_cols = global_datetime.get("columns", [])
    time_format = global_datetime.get("timeformat")
    
    if not datetime_cols:
        return df
    
    for col in datetime_cols:
        if col in df.columns:
            # Convert to string first to ensure we have the original format
            original_values = df[col].astype(str)
            
            # Strip microseconds 
            original_values = original_values.str.replace(r'\.\d+$', '', regex=True)
            
            if time_format:
                # Try with specified format first
                parsed = pd.to_datetime(original_values, errors="coerce", format=time_format)
                
                # For values that failed with format, try without format (preserves time components)
                failed_mask = parsed.isna() & df[col].notna()
                if failed_mask.any():
                    failed_values = original_values.loc[failed_mask]
                    parsed_failed = pd.to_datetime(failed_values, errors="coerce")
                    parsed.loc[failed_mask] = parsed_failed
            else:
                # No format specified - let pandas infer (handles both date and datetime)
                # Date-only values will automatically get 00:00:00
                parsed = pd.to_datetime(original_values, errors="coerce")
            
            # Assign the parsed values
            df[col] = parsed
            
    return df

def map_pids_to_ints(df: pd.DataFrame, subject_id_mapping: Dict[str, int]) -> pd.DataFrame:
    """Map PIDs to integers, with robust diagnostics and safe casting."""
    col = SUBJECT_ID
    
    # Validate non-scalar values
    non_scalar_mask = ~df[col].apply(pd.api.types.is_scalar)
    if non_scalar_mask.any():
        idx = df.index[non_scalar_mask][0]
        val = df.at[idx, col]
        raise TypeError(
            f"Non-scalar SUBJECT_ID at row {idx}, value: {repr(val)}, type: {type(val).__name__}"
        )
    
    # Map and validate unmapped values
    mapped = df[col].map(subject_id_mapping)
    unmapped_mask = mapped.isna() & df[col].notna()
    if unmapped_mask.any():
        bad_idx = df.index[unmapped_mask][:20].tolist()
        sample_vals = df.loc[bad_idx, col].tolist()
        raise KeyError(
            f"SUBJECT_ID values not found in mapping. Sample indices: {bad_idx}, values: {sample_vals}"
        )
    
    # Assign, drop NaNs, and convert to int
    df = df.copy()
    df[col] = mapped
    df = df.dropna(subset=[col])
    df[col] = pd.to_numeric(df[col], errors="raise").astype("Int64")
    
    if df[col].isna().any():
        bad_idx = df.index[df[col].isna()][:20].tolist()
        raise ValueError(f"Unexpected NA after conversion. Example indices: {bad_idx}")
    
    df[col] = df[col].astype(int)
    return df



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data."""
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
    if TIMESTAMP in df.columns:
        required_cols.append(TIMESTAMP)

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
