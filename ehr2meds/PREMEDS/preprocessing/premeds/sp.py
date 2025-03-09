from typing import Dict, Optional, Tuple

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import SUBJECT_ID
from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import (
    clean_data, convert_numeric_columns, fill_missing_values, map_pids_to_ints,
    prefix_codes, select_and_rename_columns)
from ehr2meds.PREMEDS.preprocessing.premeds.helpers import (
    create_events_dataframe, finalize_previous_patient,
    initialize_patient_state, prepare_last_patient_info,
    preprocess_admissions_df, process_patient_events)


class ConceptProcessor:
    """Handles the processing of medical concepts"""

    @staticmethod
    def process(
        df: pd.DataFrame, concept_config: dict, subject_id_mapping: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Main method for processing a single concept's data
        """
        df = select_and_rename_columns(df, concept_config.get("rename_columns", {}))
        if concept_config.get("fillna"):
            df = fill_missing_values(df, concept_config.fillna)

        df = prefix_codes(df, concept_config.get("code_prefix", None))

        df = convert_numeric_columns(df, concept_config)
        df = map_pids_to_ints(df, subject_id_mapping)
        df = clean_data(df)

        return df

    @staticmethod
    def process_adt_admissions(
        df: pd.DataFrame,
        admissions_config: dict,
        subject_id_mapping: Dict[str, int],
        last_patient_data: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, Optional[dict]]:
        """
        Process ADT admissions data to create admission/discharge events and department transfers.
        Handles patients that span across chunks.

        Args:
            df: Input DataFrame
            admissions_config: Configuration for admissions processing
            subject_id_mapping: Mapping from original IDs to integer IDs
            last_patient_data: Data from the last patient in previous chunk

        Returns:
            Tuple containing:
            - Processed DataFrame with events
            - Data for the last patient if it's incomplete (spans to next chunk)
        """
        # Preprocess the dataframe
        df = preprocess_admissions_df(df, admissions_config, subject_id_mapping)

        # Initialize state from last chunk if available
        patient_state = initialize_patient_state(last_patient_data)

        # Process all patients and generate events
        events = []

        for subject_id, patient_df in df.groupby(SUBJECT_ID):
            # Handle patient transition if needed
            if (
                patient_state["current_patient_id"] is not None
                and subject_id != patient_state["current_patient_id"]
            ):
                finalize_previous_patient(events, patient_state)

            # Set current patient
            patient_state["current_patient_id"] = subject_id

            # Process this patient's events
            process_patient_events(
                subject_id, patient_df, patient_state, events, admissions_config
            )

        # Convert events to DataFrame and sort
        result_df = create_events_dataframe(events)

        # Prepare data for the last patient if their events might continue in next chunk
        last_patient_info = prepare_last_patient_info(patient_state)

        return result_df, last_patient_info
