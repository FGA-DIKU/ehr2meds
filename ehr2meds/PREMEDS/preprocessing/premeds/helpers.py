from typing import Dict, Optional
from dataclasses import dataclass

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import (
    ADMISSION_ADT,
    CODE,
    DEPT_PREFIX,
    DISCHARGE_ADT,
    MOVE_ADT,
    SUBJECT_ID,
    TIMESTAMP,
)
from ehr2meds.PREMEDS.preprocessing.io.config import AdmissionsConfig
from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import (
    map_pids_to_ints,
    select_and_rename_columns,
)


@dataclass
class PatientState:
    """State tracking for a patient during processing."""

    current_patient_id: Optional[int] = None
    admission_start: Optional[dict] = None
    last_transfer: Optional[dict] = None

    def reset_admission_data(self):
        """Reset admission-related state."""
        self.admission_start = None
        self.last_transfer = None


def create_event(subject_id: int, timestamp, code: str) -> dict:
    """Create a standardized event dictionary."""
    return {
        SUBJECT_ID: subject_id,
        TIMESTAMP: timestamp,
        CODE: code,
    }


def preprocess_admissions_df(
    df: pd.DataFrame, config: AdmissionsConfig, subject_id_mapping: Dict[str, int]
) -> pd.DataFrame:
    """Preprocess the admissions DataFrame."""
    # Select and rename columns
    df = select_and_rename_columns(df, config.rename_columns)

    # Map subject_id to integers
    if SUBJECT_ID in df.columns:
        df = map_pids_to_ints(df, subject_id_mapping)

    # Sort by patient and timestamp
    return df.sort_values([SUBJECT_ID, config.timestamp_in_column])


def initialize_patient_state(last_patient_data: Optional[dict]) -> PatientState:
    """Initialize the current patient state from last chunk if available."""
    if last_patient_data:
        return PatientState(
            current_patient_id=last_patient_data["subject_id"],
            admission_start=last_patient_data["admission_start"],
            last_transfer=last_patient_data["last_transfer"],
        )
    else:
        return PatientState()


def finalize_previous_patient(
    events: list, patient_state: PatientState, timestamp_out_column: str
) -> None:
    """Add discharge event for previous patient if needed."""
    if (
        patient_state.admission_start is not None
        and patient_state.last_transfer is not None
        and timestamp_out_column in patient_state.last_transfer
    ):
        events.append(
            create_event(
                patient_state.current_patient_id,
                patient_state.last_transfer[timestamp_out_column],
                DISCHARGE_ADT,
            )
        )

    # Reset admission data
    patient_state.reset_admission_data()


def process_patient_events(
    subject_id: int,
    patient_df: pd.DataFrame,
    patient_state: PatientState,
    events: list,
    config: AdmissionsConfig,
) -> None:
    """Process events for a single patient."""
    for _, row in patient_df.iterrows():
        if config.type_column is None:
            # Handle simple admission with timestamp_out (no transfers)
            if (
                config.timestamp_out_column in patient_df.columns
                and config.timestamp_in_column in patient_df.columns
            ):
                handle_simple_admission_event(
                    subject_id, row, patient_state, events, config
                )
            continue

        event_type = row[config.type_column].lower()
        timestamp_in = row[config.timestamp_in_column]
        dept = (
            row.get(config.section_column)
            if config.section_column in patient_df.columns
            else None
        )

        if event_type == config.admission_event_type:
            handle_admission_event(
                subject_id, timestamp_in, dept, row, patient_state, events, config
            )
        elif (
            event_type == config.transfer_event_type
            and patient_state.admission_start is not None
        ):
            handle_transfer_event(
                subject_id, timestamp_in, dept, row, patient_state, events, config
            )
        else:
            print(f"Skipping event: {event_type}")
            break


def handle_admission_event(
    subject_id: int,
    timestamp_in,
    dept: str,
    row: pd.Series,
    patient_state: PatientState,
    events: list,
    config: AdmissionsConfig,
) -> None:
    """Handle a new admission event."""
    # If there was a previous admission, add discharge at last transfer
    if (
        patient_state.admission_start is not None
        and patient_state.last_transfer is not None
    ):
        events.append(
            create_event(
                subject_id,
                patient_state.last_transfer[config.timestamp_out_column],
                DISCHARGE_ADT,
            )
        )

    # Start new admission
    patient_state.admission_start = row

    # Add admission event
    events.append(create_event(subject_id, timestamp_in, ADMISSION_ADT))

    # Add department code if department exists
    if dept is not None:
        events.append(create_event(subject_id, timestamp_in, f"{DEPT_PREFIX}{dept}"))


def handle_simple_admission_event(
    subject_id: int,
    row: pd.Series,
    patient_state: PatientState,
    events: list,
    config: AdmissionsConfig,
) -> None:
    """Handle a simple admission event with timestamp_out (no transfers)."""
    timestamp_in = row[config.timestamp_in_column]
    timestamp_out = row[config.timestamp_out_column]
    dept = row.get(config.section_column)

    # If there was a previous admission, add discharge
    if (
        patient_state.admission_start is not None
        and patient_state.last_transfer is not None
    ):
        events.append(
            create_event(
                subject_id,
                patient_state.last_transfer[config.timestamp_out_column],
                DISCHARGE_ADT,
            )
        )

    # Start new admission
    patient_state.admission_start = row

    # Add admission event
    events.append(create_event(subject_id, timestamp_in, ADMISSION_ADT))

    # Add department code if department exists
    if dept is not None:
        events.append(create_event(subject_id, timestamp_in, f"{DEPT_PREFIX}{dept}"))

    # Add discharge event immediately
    events.append(create_event(subject_id, timestamp_out, DISCHARGE_ADT))

    # Reset admission data since this is a complete admission-discharge cycle
    patient_state.reset_admission_data()


def handle_transfer_event(
    subject_id: int,
    timestamp_in,
    dept: str,
    row: pd.Series,
    patient_state: PatientState,
    events: list,
    config: AdmissionsConfig,
) -> None:
    """Handle a transfer event."""
    # Record transfer if configured to do so
    if config.save_adm_move:
        events.append(create_event(subject_id, timestamp_in, MOVE_ADT))

    # Add department code for the new department if department exists
    if dept is not None:
        events.append(create_event(subject_id, timestamp_in, f"{DEPT_PREFIX}{dept}"))

    # Update last transfer
    patient_state.last_transfer = row


def create_events_dataframe(events: list) -> pd.DataFrame:
    """Convert events list to DataFrame and sort."""
    result_df = pd.DataFrame(events)

    # Sort by patient and timestamp if not empty
    if not result_df.empty:
        result_df = result_df.sort_values([SUBJECT_ID, TIMESTAMP])

    return result_df


def prepare_last_patient_info(patient_state: PatientState) -> Optional[dict]:
    """Prepare information about the last patient for the next chunk."""
    if (
        patient_state.current_patient_id is not None
        and patient_state.admission_start is not None
    ):
        return {
            SUBJECT_ID: patient_state.current_patient_id,
            "admission_start": patient_state.admission_start,
            "last_transfer": patient_state.last_transfer,
            "events": [],  # Will be populated if this is the final chunk
        }
    return None


def add_discharge_to_last_patient(
    last_patient_data: Optional[dict], timestamp_out_column: str
) -> pd.DataFrame:
    """
    Process any remaining last patient data by adding a discharge event.

    Args:
        last_patient_data: Dictionary containing information about the last patient
                          from the previous chunk processing
        config: Configuration for admissions processing

    Returns:
        pd.DataFrame: DataFrame containing the final discharge event or empty DataFrame
                     if there's no remaining patient data
    """
    if last_patient_data is None or last_patient_data["last_transfer"] is None:
        return pd.DataFrame()  # No data to process

    # Check if the timestamp_out column exists in the last transfer data
    if timestamp_out_column not in last_patient_data["last_transfer"]:
        return pd.DataFrame()  # No timestamp_out data available

    # Create discharge event for the last patient
    events = last_patient_data.get("events", [])
    events.append(
        create_event(
            last_patient_data[SUBJECT_ID],
            last_patient_data["last_transfer"][timestamp_out_column],
            DISCHARGE_ADT,
        )
    )

    # Convert to DataFrame
    final_df = pd.DataFrame(events)
    if not final_df.empty:
        final_df = final_df.sort_values([SUBJECT_ID, TIMESTAMP])

    return final_df
