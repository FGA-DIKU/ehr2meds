import pandas as pd
from ehr2meds.preMEDS.constants import (
    MANDATORY_COLUMNS,
    ROW_INDEX,
    SUBJECT_ID,
)
from typing import Dict


def add_row_idx(df: pd.DataFrame, start: int = 0) -> pd.DataFrame:
    """Add a stable, contiguous source-row index to a preMEDS chunk."""
    df[ROW_INDEX] = range(start, start + len(df))
    return df


def check_columns(df: pd.DataFrame, columns_map: dict):
    """Check if all columns in columns_map are present in df."""
    missing_columns = set(columns_map.keys()) - set(df.columns)
    if missing_columns:
        available_columns = pd.DataFrame({"Available Columns": sorted(df.columns)})
        requested_columns = pd.DataFrame({"Requested Columns": sorted(columns_map.keys())})
        error_msg = f"\nMissing columns: {sorted(missing_columns)}\n\n"
        error_msg += "Columns comparison:\n"
        error_msg += f"{pd.concat([available_columns, requested_columns], axis=1).to_string()}"
        raise ValueError(error_msg)


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


def map_pids_to_ints(df: pd.DataFrame, subject_id_mapping: Dict[str, int]) -> pd.DataFrame:
    """Map string patient IDs to integers; keep only IDs that are in the mapping."""
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(object).astype(str)

    df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
    if df[SUBJECT_ID].isna().any():
        missing_ids = df[SUBJECT_ID][df[SUBJECT_ID].isna()].unique()
        print(f"Found {len(missing_ids)} subject IDs in the data that are not in the mapping. These IDs will be dropped")
    df = df.dropna(subset=[SUBJECT_ID], how="any")
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data."""
    # Clean data
    if all(col in df.columns for col in MANDATORY_COLUMNS):
        df = df.dropna(subset=MANDATORY_COLUMNS, how="any")

    # row_idx is always unique, so don't consider that column
    columns_to_check = [col for col in df.columns if col != ROW_INDEX]

    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates(columns_to_check)
    n_after = len(df)
    if n_before > n_after:
        print(f"Dropped {n_before - n_after} rows from {columns_to_check} due to duplicates")
    return df


def apply_value_map(df: pd.DataFrame, value_map_cfg: dict) -> pd.DataFrame:
    """Map column values using inline config mapping. Unmapped values become NaN."""
    n_before = len(df)
    for col, mapping in value_map_cfg.items():
        df.replace({col: mapping}, inplace=True)
    n_after = len(df)
    if n_before > n_after:
        print(f"Dropped {n_before - n_after} rows from {col} due to value mapping")
    return df


def validate_subject_id(df: pd.DataFrame) -> None:
    """Checks that the subject_id column exists and is an integer"""
    if SUBJECT_ID not in df.columns:
        raise ValueError(f"Missing required column: {SUBJECT_ID}")
    if not pd.api.types.is_integer_dtype(df[SUBJECT_ID]):
        raise ValueError(
            f"{SUBJECT_ID} column must be of integer type\n\
                Hint: Use the subject_id_mapping configuration to map string IDs to integers."
        )


def remove_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timezone-aware datetime columns to timezone-naive UTC."""
    for col in df.select_dtypes(include=["datetimetz"]).columns:
        df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)

    return df
