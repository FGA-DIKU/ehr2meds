from typing import Dict, List, Optional, Tuple

import pandas as pd

from MEDS_preprocess.preprocessors.constants import CODE, MANDATORY_COLUMNS, SUBJECT_ID
from MEDS_preprocess.preprocessors.helpers import DataHandler
from MEDS_preprocess.preprocessors.concept_processor_utils import select_and_rename_columns, process_codes
    

class RegisterConceptProcessor:
    @staticmethod
    def process_register_concept(
        df: pd.DataFrame,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        data_handler: "DataHandler",
        register_sp_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process the register concepts."""
        # Step 1: Initial processing
        df = RegisterConceptProcessor._process_initial_register_data(df, concept_config)

        # Step 2: Apply secondary mapping if needed
        df = ConceptProcessor._apply_secondary_mapping(df, concept_config, data_handler)

        # Step 3: Convert numeric columns
        df = ConceptProcessor._convert_numeric_columns(df, concept_config)

        # Step 4: Apply main mapping and register mapping
        df = RegisterConceptProcessor._apply_main_and_register_mapping(
            df, concept_config, data_handler, register_sp_mapping
        )

        # Step 5: Process codes (unroll or prefix)
        df = RegisterConceptProcessor._process_register_codes(df, concept_config)

        # Step 6: Final cleanup and mapping
        df = ConceptProcessor._map_and_clean_data(df, subject_id_mapping)

        return df
        #
    @staticmethod
    def _process_initial_register_data(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Handle initial data processing steps."""
        # Select and rename columns
        df = select_and_rename_columns(
            df, concept_config.get("columns_map", {})
        )

        # Combine datetime columns if needed
        df = RegisterConceptProcessor._combine_datetime_columns(df, concept_config)

        return df
    
    @staticmethod
    def _apply_main_and_register_mapping(
        df: pd.DataFrame,
        concept_config: dict,
        data_handler: "DataHandler",
        register_sp_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply main mapping and register mapping to get final subject IDs."""
        if "main_mapping" not in concept_config:
            return df

        mapping_cfg = concept_config["main_mapping"]
        mapping_df = ConceptProcessor._load_mapping_file(mapping_cfg, data_handler)

        # Apply main mapping
        df = pd.merge(
            df,
            mapping_df,
            left_on=mapping_cfg.get("left_on"),
            right_on=mapping_cfg.get("right_on"),
            how="inner",
        )

        # Apply register mapping if possible
        if "PID" in df.columns and "PID" in register_sp_mapping.columns:
            df = pd.merge(df, register_sp_mapping, on="PID", how="inner")
            if SUBJECT_ID in register_sp_mapping.columns:
                df = df.drop(columns=[mapping_cfg.get("left_on"), "PID"])
                df = df.rename(columns={"SP_HASH": SUBJECT_ID})
        else:
            df = df.drop(columns=[mapping_cfg.get("left_on")])

        return df

    @staticmethod
    def _process_register_codes(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
        """Process codes through unrolling or adding prefixes."""
        if "unroll_columns" in concept_config:
            processed_dfs = ConceptProcessor._unroll_columns(df, concept_config)
            return pd.concat(processed_dfs, ignore_index=True) if processed_dfs else df

        # Add code prefix if specified
        code_prefix = concept_config.get("code_prefix", "")
        if code_prefix and CODE in df.columns:
            df[CODE] = code_prefix + df[CODE].astype(str)

        return df
    
    @staticmethod
    def _combine_datetime_columns(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Combine date and time columns into datetime columns."""
        if "combine_datetime" in concept_config:
            for target_col, date_time_cols in concept_config[
                "combine_datetime"
            ].items():
                date_col = date_time_cols.get("date_col")
                time_col = date_time_cols.get("time_col")
                if date_col in df.columns and time_col in df.columns:
                    df[target_col] = pd.to_datetime(
                        df[date_col].astype(str) + " " + df[time_col].astype(str),
                        errors="coerce",
                    )
                    # Drop original columns if requested
                    if date_time_cols.get("drop_original", True):
                        df = df.drop(columns=[date_col, time_col])
        return df

class ConceptProcessor:
    """Handles the processing of medical concepts"""
    @staticmethod
    def process_concept(
        df: pd.DataFrame, concept_config: dict, subject_id_mapping: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Main method for processing a single concept's data
        """
        df = select_and_rename_columns(
            df, concept_config.get("columns_map", {})
        )
        df = process_codes(df, concept_config)
        df = ConceptProcessor._convert_and_clean_data(
            df, concept_config, subject_id_mapping
        )

        return df

    @staticmethod
    def _apply_secondary_mapping(
        df: pd.DataFrame, concept_config: dict, data_handler: "DataHandler"
    ) -> pd.DataFrame:
        """Apply secondary mapping (e.g., vnr to drug name)."""
        if "secondary_mapping" not in concept_config:
            return df

        mapping_cfg = concept_config["secondary_mapping"]
        mapping_df = ConceptProcessor._load_mapping_file(mapping_cfg, data_handler)

        code_col = mapping_cfg.get("code_column")
        if code_col:
            df = ConceptProcessor._apply_code_mapping(
                df, mapping_df, mapping_cfg, code_col
            )
        else:
            df = ConceptProcessor._apply_simple_mapping(df, mapping_df, mapping_cfg)

        return df

    @staticmethod
    def _apply_code_mapping(
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

    @staticmethod
    def _apply_simple_mapping(
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


    @staticmethod
    def _convert_numeric_columns(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Convert specified columns to numeric type."""
        numeric_cols = concept_config.get("numeric_columns", [])
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _unroll_columns(df: pd.DataFrame, concept_config: dict) -> List[pd.DataFrame]:
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

    @staticmethod
    def _map_and_clean_data(df: pd.DataFrame, subject_id_mapping: dict) -> pd.DataFrame:
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

    @staticmethod
    def _load_mapping_file(
        mapping_cfg: dict, data_handler: "DataHandler"
    ) -> pd.DataFrame:
        """Load a mapping file based on configuration."""
        filename = mapping_cfg.get("filename")
        # If data_handler is provided, use it to load from datastore
        return data_handler.load_pandas({"filename": filename})


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
        df = select_and_rename_columns(
            df, admissions_config.get("columns_map", {})
        )
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

