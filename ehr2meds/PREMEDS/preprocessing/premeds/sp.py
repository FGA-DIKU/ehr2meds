from typing import Dict, Optional, Tuple

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import CODE, MANDATORY_COLUMNS, SUBJECT_ID
from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import (
    process_codes,
    select_and_rename_columns,
)


class ConceptProcessor:
    """Handles the processing of medical concepts"""

    @staticmethod
    def process_concept(
        df: pd.DataFrame, concept_config: dict, subject_id_mapping: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Main method for processing a single concept's data
        """
        df = select_and_rename_columns(df, concept_config.get("columns_map", {}))
        df = process_codes(df, concept_config)
        df = ConceptProcessor._convert_and_clean_data(
            df, concept_config, subject_id_mapping
        )

        return df

    @staticmethod
    def _convert_and_clean_data(
        df: pd.DataFrame, concept_config: dict, subject_id_mapping: dict
    ) -> pd.DataFrame:
        """Convert numeric columns, map subject IDs, and clean the data."""
        # Convert numeric columns
        numeric_cols = concept_config.get("numeric_columns", [])
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Map subject_id if available
        if SUBJECT_ID in df.columns:
            df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
            # Drop rows where mapping failed
            df = df.dropna(subset=[SUBJECT_ID])
            # Convert to integer
            df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)

        # Clean data
        if all(col in df.columns for col in MANDATORY_COLUMNS):
            df = df.dropna(subset=MANDATORY_COLUMNS, how="any")
        df = df.drop_duplicates()

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
        # First select and rename columns
        df = select_and_rename_columns(df, admissions_config.get("rename_columns", {}))
        # Map subject_id
        if SUBJECT_ID in df.columns:
            df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
            df = df.dropna(subset=[SUBJECT_ID])
            df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)

        # Sort by patient and timestamp
        df = df.sort_values([SUBJECT_ID, "timestamp_in"])

        # Initialize lists to store the processed events
        events = []

        # If we have data from last chunk's patient, initialize with that
        current_patient_id = None
        admission_start = None
        last_transfer = None

        if last_patient_data:
            current_patient_id = last_patient_data["subject_id"]
            admission_start = last_patient_data["admission_start"]
            last_transfer = last_patient_data["last_transfer"]

        # Process each patient's events
        for subject_id, patient_df in df.groupby(SUBJECT_ID):
            # If this is a new patient and we had a previous patient's data,
            # finalize the previous patient's events
            if current_patient_id is not None and subject_id != current_patient_id:
                if admission_start is not None and last_transfer is not None:
                    events.append(
                        {
                            SUBJECT_ID: current_patient_id,
                            "timestamp": last_transfer["timestamp_out"],
                            CODE: "DISCHARGE_ADT",
                        }
                    )

                # Reset for new patient
                admission_start = None
                last_transfer = None

            current_patient_id = subject_id
            for _, row in patient_df.iterrows():
                event_type = row["type"]
                dept = row["section"]
                timestamp_in = row["timestamp_in"]

                if event_type.lower() == "indlaeggelse":
                    # If there was a previous admission, add discharge at last transfer
                    if admission_start is not None and last_transfer is not None:
                        events.append(
                            {
                                SUBJECT_ID: subject_id,
                                "timestamp": last_transfer["timestamp_out"],
                                CODE: "DISCHARGE_ADT",
                            }
                        )

                    # Start new admission
                    admission_start = row
                    events.append(
                        {
                            SUBJECT_ID: subject_id,
                            "timestamp": timestamp_in,
                            CODE: "ADMISSION_ADT",
                        }
                    )

                    # Add department code
                    events.append(
                        {
                            SUBJECT_ID: subject_id,
                            "timestamp": timestamp_in,
                            CODE: f"AFSNIT_ADT_{dept}",
                        }
                    )

                elif event_type.lower() == "flyt ind" and admission_start is not None:
                    # Record transfer
                    if admissions_config.get("save_adm_move", True):
                        events.append(
                            {
                                SUBJECT_ID: subject_id,
                                "timestamp": timestamp_in,
                                CODE: "ADM_move",
                            }
                        )
                    # Add department code for the new department
                    events.append(
                        {
                            SUBJECT_ID: subject_id,
                            "timestamp": timestamp_in,
                            CODE: f"AFSNIT_ADT_{dept}",
                        }
                    )
                    last_transfer = row

        # Convert events to DataFrame
        result_df = pd.DataFrame(events)
        # Sort by patient and timestamp
        if not result_df.empty:
            result_df = result_df.sort_values([SUBJECT_ID, "timestamp"])

        # Prepare data for the last patient if their events might continue in next chunk
        last_patient_info = None
        if current_patient_id is not None and admission_start is not None:
            last_patient_info = {
                "subject_id": current_patient_id,
                "admission_start": admission_start,
                "last_transfer": last_transfer,
                "events": [],  # Will be populated if this is the final chunk
            }

        return result_df, last_patient_info
