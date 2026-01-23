import logging
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
    """Factorize the subject_id column into an integer mapping.

    Args:
        df: DataFrame containing SUBJECT_ID column

    Returns:
        Tuple[pd.DataFrame, Dict[str, int]]:
            - DataFrame with integer subject IDs
            - Mapping from original IDs to integer IDs

    Example:
        Input df[SUBJECT_ID]: ['A', 'B', 'C', 'D']
        Output mapping: {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    """
    # Get unique values and create sequential mapping
    unique_vals = df[SUBJECT_ID].unique()
    hash_to_int_map = {
        val: idx + 2 for idx, val in enumerate(sorted(unique_vals))
    }  # +2 to prevent subject ids being read in as binary.

    print(df.head())

    # Apply mapping to DataFrame
    try:
        logger.debug(f"Mapping {len(df)} rows with {len(hash_to_int_map)} unique subject IDs")
        original_subject_ids = df[SUBJECT_ID].copy()
        
        # Check for problematic values before mapping (e.g., lists, dicts, etc.)
        problematic_indices = []
        for idx, val in original_subject_ids.items():
            try:
                # Try to use the value as a dictionary key to see if it's hashable
                _ = hash_to_int_map.get(val)
            except (TypeError, ValueError) as e:
                problematic_indices.append(idx)
                logger.warning(
                    f"Row {idx} has problematic SUBJECT_ID value (type: {type(val).__name__}, value: {val}). "
                    f"Full row:\n{df.loc[idx].to_dict()}"
                )
        
        if problematic_indices:
            logger.error(
                f"Found {len(problematic_indices)} rows with non-hashable SUBJECT_ID values. "
                f"Indices: {problematic_indices[:10]}"
            )
        
        # Try mapping and catch the exact error
        print(hash_to_int_map)
        try:
            df.loc[:, SUBJECT_ID] = df[SUBJECT_ID].map(hash_to_int_map)
        except (TypeError, ValueError) as map_error:
            logger.error(f"Mapping failed with error: {str(map_error)}")
            logger.error("Finding exact problematic row(s)...")
            
            # Try mapping row by row to find the exact failure
            mapped_series = pd.Series(index=df.index, dtype=object)
            failed_rows = []
            
            for idx in df.index:
                val = original_subject_ids.loc[idx]
                try:
                    # First check if value is hashable
                    try:
                        hash(val)
                    except TypeError:
                        failed_rows.append(idx)
                        logger.error(
                            f"\n{'='*80}\n"
                            f"FAILED ROW {idx} (Unhashable type):\n"
                            f"  SUBJECT_ID value: {repr(val)}\n"
                            f"  SUBJECT_ID type: {type(val).__name__}\n"
                            f"  Full row data:\n{df.loc[idx].to_dict()}\n"
                            f"{'='*80}\n"
                        )
                        continue
                    
                    # Try to map this specific value
                    mapped_val = hash_to_int_map.get(val)
                    mapped_series.loc[idx] = mapped_val
                except Exception as row_error:
                    failed_rows.append(idx)
                    logger.error(
                        f"\n{'='*80}\n"
                        f"FAILED ROW {idx}:\n"
                        f"  SUBJECT_ID value: {repr(val)}\n"
                        f"  SUBJECT_ID type: {type(val).__name__}\n"
                        f"  Error: {str(row_error)}\n"
                        f"  Full row data:\n{df.loc[idx].to_dict()}\n"
                        f"{'='*80}\n"
                    )
            
            if failed_rows:
                logger.error(f"Total failed rows: {len(failed_rows)}. Indices: {failed_rows}")
                raise ValueError(
                    f"Mapping failed for {len(failed_rows)} row(s). "
                    f"First failed row index: {failed_rows[0]}. "
                    f"See logs above for details."
                )
            else:
                # If we got here, the error was something else - re-raise original
                raise map_error
        
        # Check for unmapped values (NaN after mapping)
        unmapped_count = df[SUBJECT_ID].isna().sum()
        if unmapped_count > 0:
            unmapped_mask = df[SUBJECT_ID].isna()
            unmapped_original_values = original_subject_ids[unmapped_mask].unique()
            unmapped_indices = df[unmapped_mask].index.tolist()
            
            logger.warning(
                f"Found {unmapped_count} rows with unmapped subject IDs. "
                f"Sample unmapped values: {list(unmapped_original_values[:10])}"
            )
            
            # Print first few failing rows
            for idx in unmapped_indices[:5]:
                logger.warning(
                    f"Row {idx} failed to map. SUBJECT_ID value: {original_subject_ids.loc[idx]}, "
                    f"Type: {type(original_subject_ids.loc[idx]).__name__}. "
                    f"Full row:\n{df.loc[idx].to_dict()}"
                )
        
        logger.debug(f"Converting SUBJECT_ID to int. Current dtype: {df[SUBJECT_ID].dtype}")
        df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)
        logger.debug(f"Successfully converted SUBJECT_ID to int. New dtype: {df[SUBJECT_ID].dtype}")
    except KeyError as e:
        logger.error(f"SUBJECT_ID column not found in dataframe. Available columns: {list(df.columns)}")
        raise
    except (TypeError, ValueError) as e:
        # Try to identify the problematic row by iterating
        logger.error(f"Error during mapping: {str(e)}")
        logger.error("Attempting to identify problematic rows...")
        
        problematic_rows = []
        for idx, val in original_subject_ids.items():
            try:
                # Try to look up in mapping
                _ = hash_to_int_map.get(val)
            except Exception as row_error:
                problematic_rows.append(idx)
                logger.error(
                    f"Row {idx} has problematic SUBJECT_ID: {val} (type: {type(val).__name__}). "
                    f"Error: {str(row_error)}. "
                    f"Full row:\n{df.loc[idx].to_dict()}"
                )
        
        if problematic_rows:
            logger.error(f"Found {len(problematic_rows)} problematic rows: {problematic_rows[:20]}")
        else:
            logger.error("Could not identify specific problematic rows. Showing first few rows:")
            for idx in df.index[:5]:
                logger.error(f"Row {idx}: SUBJECT_ID = {original_subject_ids.loc[idx]} (type: {type(original_subject_ids.loc[idx]).__name__})")
        
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during subject ID mapping/conversion. "
            f"DataFrame shape: {df.shape}, "
            f"SUBJECT_ID dtype: {df[SUBJECT_ID].dtype if SUBJECT_ID in df.columns else 'N/A'}, "
            f"Error: {str(e)}"
        )
        # Try to find problematic rows even in unexpected errors
        try:
            for idx in df.index[:10]:
                val = original_subject_ids.loc[idx]
                logger.error(
                    f"Row {idx} sample: SUBJECT_ID = {val} (type: {type(val).__name__})"
                )
        except Exception:
            pass
        raise
    
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


def map_pids_to_ints(
    df: pd.DataFrame, subject_id_mapping: Dict[str, int]
) -> pd.DataFrame:
    """Map PIDs to integers."""
    df.loc[:, SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
    df = df.dropna(subset=[SUBJECT_ID])
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(float).astype(int)
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
