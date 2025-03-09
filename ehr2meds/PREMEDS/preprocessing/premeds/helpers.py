from typing import Dict, Optional

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import CODE, SUBJECT_ID, TIMESTAMP
from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import (
    select_and_rename_columns,
)


def preprocess_admissions_df(
    df: pd.DataFrame, admissions_config: dict, subject_id_mapping: Dict[str, int]
) -> pd.DataFrame:
    """Preprocess the admissions DataFrame."""
    # Select and rename columns
    df = select_and_rename_columns(df, admissions_config.get("rename_columns", {}))

    # Map subject_id to integers
    if SUBJECT_ID in df.columns:
        df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
        df = df.dropna(subset=[SUBJECT_ID])
        df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)

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
                CODE: "DISCHARGE_ADT",
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

        if event_type == "indlæggelse":
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
                CODE: "DISCHARGE_ADT",
            }
        )

    # Start new admission
    patient_state["admission_start"] = row

    # Add admission event
    events.append(
        {
            SUBJECT_ID: subject_id,
            TIMESTAMP: timestamp_in,
            CODE: "ADMISSION_ADT",
        }
    )

    # Add department code
    events.append(
        {
            SUBJECT_ID: subject_id,
            TIMESTAMP: timestamp_in,
            CODE: f"AFSNIT_ADT_{dept}",
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
                CODE: "ADM_move",
            }
        )

    # Add department code for the new department
    events.append(
        {
            SUBJECT_ID: subject_id,
            TIMESTAMP: timestamp_in,
            CODE: f"AFSNIT_ADT_{dept}",
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
            CODE: "DISCHARGE_ADT",
        }
    )

    # Convert to DataFrame
    final_df = pd.DataFrame(events)
    if not final_df.empty:
        final_df = final_df.sort_values([SUBJECT_ID, TIMESTAMP])

    return final_df
