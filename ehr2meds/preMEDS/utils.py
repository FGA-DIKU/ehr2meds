import pandas as pd
from ehr2meds.preMEDS.constants import (
    CODE,
    MANDATORY_COLUMNS,
    SUBJECT_ID,
    TIMESTAMP,
)
from typing import Dict, List, Tuple


def select_and_rename_columns(df: pd.DataFrame, columns_map: dict) -> pd.DataFrame:
    """Select and rename columns based on columns_map."""
    check_columns(df, columns_map)
    df = df[list(columns_map.keys())]
    df = df.rename(columns=columns_map)
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
    # Convert to string to handle any array-like values
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(object).astype(str)

    # Get unique values and create sequential mapping
    unique_vals = df[SUBJECT_ID].unique()
    hash_to_int_map = {
        val: int(idx + 2) for idx, val in enumerate(sorted(unique_vals))
    }  # +2 to prevent subject ids being read in as binary.

    # Convert to object dtype before mapping to allow integer assignment
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(object)
    # Map to integers
    mapped = df[SUBJECT_ID].map(hash_to_int_map)
    # Drop rows where mapping failed (NaN values) before converting to int
    mask = mapped.notna()
    df = df.loc[mask].copy()
    # Create a new Series with int64 dtype explicitly
    df[SUBJECT_ID] = pd.Series(mapped.loc[mask].values, dtype="int64", index=df.index)
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
    # Convert to object dtype to allow integer assignment after mapping
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(str)
    # Map to integers and convert to int
    df.loc[:, SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
    df = df.dropna(subset=[SUBJECT_ID])
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)
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


def convert_timestamp_columns(
    df: pd.DataFrame, names: List[str], format: str
) -> pd.DataFrame:
    """Convert timestamps to global format."""
    for name in names:
        if name in df.columns:
            df[name] = pd.to_datetime(df[name]).dt.strftime(format)
    return df


def apply_value_map(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    """Map column values using inline config mapping. Unmapped values become NaN."""
    for col, mapping in concept_config.get("value_map", {}).items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def normalize_columns(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    """Normalize column values by casting to int then str, getting rid of leading zeros."""
    for col in concept_config.get("normalize_columns", []):
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce").astype("Int64").astype(str)
            )
            df[col] = df[col].replace("<NA>", None)
    return df


def replace_values(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    """Apply string replacement to specified column."""
    for col, replacements in concept_config.get("replace_values", {}).items():
        if col in df.columns:
            for old, new in replacements.items():
                df[col] = df[col].str.replace(old, new, regex=False)
    return df


def pad_values(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    """Append suffix to column values that don't already contain it."""
    for col, cfg in concept_config.get("pad_values", {}).items():
        if col in df.columns:
            suffix = cfg["suffix"]
            contains = cfg.get("unless_contains", suffix)
            mask = ~df[col].astype(str).str.contains(contains, regex=False, na=False)
            df.loc[mask, col] = df.loc[mask, col].astype(str) + suffix
    return df


def apply_melt_step(df, cfg):
    # Example df
    value_cols = cfg.get("source_cols")
    target_col = cfg.get("target_name")
    prefix_col = cfg.get("prefix_col")
    prefix_map = cfg.get("prefix_map")
    id_cols = [c for c in df.columns if c not in value_cols]
    df_melted = df.melt(
        id_vars=id_cols, value_vars=value_cols, var_name="source", value_name=target_col
    )

    # Add prefix
    df_melted[prefix_col] = df_melted["source"].map(prefix_map) + df_melted[
        prefix_col
    ].astype(str)

    # # Drop columns
    cols_to_keep = list(set([prefix_col, target_col] + id_cols))
    df_melted = df_melted[cols_to_keep]
    return df_melted


def melt_table(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
    expand_map = concept_config.get("melt_table")
    if not expand_map:
        return df

    for step_cfg in expand_map:
        df = apply_melt_step(df, step_cfg)
    return df
