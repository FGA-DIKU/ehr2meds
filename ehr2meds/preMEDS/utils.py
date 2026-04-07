from typing import Dict, List, Tuple, Optional

import pandas as pd

from ehr2meds.preMEDS.constants import (
    CODE,
    MANDATORY_COLUMNS,
    SUBJECT_ID,
    TIMESTAMP,
    ADMISSION_ADT,
    ADMISSION_IND,
    DEPT_PREFIX,
    DISCHARGE_ADT,
    MOVE_ADT,
)


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
    # (can't assign integers to string dtype column?)
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(object)
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


def preprocess_admissions_df(
    df: pd.DataFrame, admissions_config: dict, subject_id_mapping: Dict[str, int]
) -> pd.DataFrame:
    """Preprocess the admissions DataFrame."""
    # Select and rename columns
    df = select_and_rename_columns(df, admissions_config.get("rename_columns", {}))

    # Map subject_id to integers
    if SUBJECT_ID in df.columns:
        df = map_pids_to_ints(df, subject_id_mapping)

    # Sort by patient and timestamp
    return df.sort_values([SUBJECT_ID, "timestamp_in"])


def initialize_patient_state(last_patient_data: Optional[dict]) -> dict:
    """Initialize the current patient state from last chunk if available."""
    if last_patient_data:
        return {
            "current_patient_id": last_patient_data["subject_id"],
            "admission_start": last_patient_data["admission_start"],
            "last_transfer": last_patient_data["last_transfer"],
        }
    else:
        return {
            "current_patient_id": None,
            "admission_start": None,
            "last_transfer": None,
        }


def finalize_previous_patient(events: list, patient_state: dict) -> None:
    """Add discharge event for previous patient if needed."""
    if (
        patient_state["admission_start"] is not None
        and patient_state["last_transfer"] is not None
    ):
        events.append(
            {
                SUBJECT_ID: patient_state["current_patient_id"],
                TIMESTAMP: patient_state["last_transfer"]["timestamp_out"],
                CODE: DISCHARGE_ADT,
            }
        )

    # Reset admission data
    patient_state["admission_start"] = None
    patient_state["last_transfer"] = None


def process_patient_events(
    subject_id: int,
    patient_df: pd.DataFrame,
    patient_state: dict,
    events: list,
    admissions_config: dict,
) -> None:
    """Process events for a single patient."""
    for _, row in patient_df.iterrows():
        event_type = row["type"].lower()
        dept = row["section"]
        timestamp_in = row["timestamp_in"]

        if event_type == ADMISSION_IND.lower():
            handle_admission_event(
                subject_id, timestamp_in, dept, row, patient_state, events
            )
        elif event_type == "flyt ind" and patient_state["admission_start"] is not None:
            handle_transfer_event(
                subject_id,
                timestamp_in,
                dept,
                row,
                patient_state,
                events,
                admissions_config,
            )


def handle_admission_event(
    subject_id: int,
    timestamp_in,
    dept: str,
    row: pd.Series,
    patient_state: dict,
    events: list,
) -> None:
    """Handle a new admission event."""
    # If there was a previous admission, add discharge at last transfer
    if (
        patient_state["admission_start"] is not None
        and patient_state["last_transfer"] is not None
    ):
        events.append(
            {
                SUBJECT_ID: subject_id,
                TIMESTAMP: patient_state["last_transfer"]["timestamp_out"],
                CODE: DISCHARGE_ADT,
            }
        )

    # Start new admission
    patient_state["admission_start"] = row

    # Add admission event
    events.append(
        {
            SUBJECT_ID: subject_id,
            TIMESTAMP: timestamp_in,
            CODE: ADMISSION_ADT,
        }
    )

    # Add department code
    events.append(
        {
            SUBJECT_ID: subject_id,
            TIMESTAMP: timestamp_in,
            CODE: f"{DEPT_PREFIX}{dept}",
        }
    )


def handle_transfer_event(
    subject_id: int,
    timestamp_in,
    dept: str,
    row: pd.Series,
    patient_state: dict,
    events: list,
    admissions_config: dict,
) -> None:
    """Handle a transfer event."""
    # Record transfer if configured to do so
    if admissions_config.get("save_adm_move", True):
        events.append(
            {
                SUBJECT_ID: subject_id,
                TIMESTAMP: timestamp_in,
                CODE: MOVE_ADT,
            }
        )

    # Add department code for the new department
    events.append(
        {
            SUBJECT_ID: subject_id,
            TIMESTAMP: timestamp_in,
            CODE: f"{DEPT_PREFIX}{dept}",
        }
    )

    # Update last transfer
    patient_state["last_transfer"] = row


def create_events_dataframe(events: list) -> pd.DataFrame:
    """Convert events list to DataFrame and sort."""
    result_df = pd.DataFrame(events)

    # Sort by patient and timestamp if not empty
    if not result_df.empty:
        result_df = result_df.sort_values([SUBJECT_ID, TIMESTAMP])

    return result_df


def prepare_last_patient_info(patient_state: dict) -> Optional[dict]:
    """Prepare information about the last patient for the next chunk."""
    if (
        patient_state["current_patient_id"] is not None
        and patient_state["admission_start"] is not None
    ):
        return {
            SUBJECT_ID: patient_state["current_patient_id"],
            "admission_start": patient_state["admission_start"],
            "last_transfer": patient_state["last_transfer"],
            "events": [],  # Will be populated if this is the final chunk
        }
    return None


def add_discharge_to_last_patient(last_patient_data: Optional[dict]) -> pd.DataFrame:
    """
    Process any remaining last patient data by adding a discharge event.

    Args:
        last_patient_data: Dictionary containing information about the last patient
                          from the previous chunk processing

    Returns:
        pd.DataFrame: DataFrame containing the final discharge event or empty DataFrame
                     if there's no remaining patient data
    """
    if last_patient_data is None or last_patient_data["last_transfer"] is None:
        return pd.DataFrame()  # No data to process

    # Create discharge event for the last patient
    events = last_patient_data.get("events", [])
    events.append(
        {
            SUBJECT_ID: last_patient_data[SUBJECT_ID],
            TIMESTAMP: last_patient_data["last_transfer"]["timestamp_out"],
            CODE: DISCHARGE_ADT,
        }
    )

    # Convert to DataFrame
    final_df = pd.DataFrame(events)
    if not final_df.empty:
        final_df = final_df.sort_values([SUBJECT_ID, TIMESTAMP])

    return final_df
