import pandas as pd
from ehr2meds.preMEDS.constants import SUBJECT_ID
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.utils import (
    apply_mapping,
    clean_data,
    convert_numeric_columns,
    create_events_dataframe,
    fill_missing_values,
    finalize_previous_patient,
    initialize_patient_state,
    map_pids_to_ints,
    prepare_last_patient_info,
    preprocess_admissions_df,
    process_patient_events,
    select_and_rename_columns,
    unroll_columns,
    convert_timestamp_columns
)
from typing import Dict, Optional, Tuple


class SPConceptProcessor:

    @staticmethod
    def process(df: pd.DataFrame, concept_config: dict, subject_id_mapping: Dict[str, int], time_stamp_dict: Optional[dict] = None) -> pd.DataFrame:
        """
        Main method for processing a single concept's data
        """
        df = select_and_rename_columns(df, concept_config.get("rename_columns", {}))
        if concept_config.get("fillna"):
            df = fill_missing_values(df, concept_config.fillna)

        if time_stamp_dict:
            print(f"Converting timestamp columns: {time_stamp_dict}")
            df = convert_timestamp_columns(df, **time_stamp_dict)

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
            if patient_state["current_patient_id"] is not None and subject_id != patient_state["current_patient_id"]:
                finalize_previous_patient(events, patient_state)

            # Set current patient
            patient_state["current_patient_id"] = subject_id

            # Process this patient's events
            process_patient_events(subject_id, patient_df, patient_state, events, admissions_config)

        # Convert events to DataFrame and sort
        result_df = create_events_dataframe(events)

        # Prepare data for the last patient if their events might continue in next chunk
        last_patient_info = prepare_last_patient_info(patient_state)

        return result_df, last_patient_info


class RegisterConceptProcessor:
    @staticmethod
    def process(
        df: pd.DataFrame,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        data_handler: "DataHandler",
        register_sp_link: pd.DataFrame,
        join_link_col: str,
        target_link_col: str,
    ) -> pd.DataFrame:
        """Process the register concepts.
        1. Select and rename columns
        2. apply columns map
        3. fill missing values
        4. combine datetime columns
        5. unroll columns (process codes)
        6. apply prefixes (process codes)
        7. convert numeric columns
        8. apply pid linking
        9. apply pid integer mapping
        10. clean data
        """
        df = select_and_rename_columns(df, concept_config.get("rename_columns", {}))
        df = RegisterConceptProcessor._apply_mappings(df, concept_config, data_handler)
        df = fill_missing_values(df, concept_config.get("fillna", {}))
        df = RegisterConceptProcessor._combine_datetime_columns(df, concept_config)

        # Apply code prefix to the original code column before other columns are unrolled
        # the unrolled columns can get their own prefixes
        df = prefix_codes(df, concept_config.get("code_prefix", None))

        df = RegisterConceptProcessor._unroll_columns(df, concept_config)

        df = convert_numeric_columns(df, concept_config)

        df = RegisterConceptProcessor._apply_sp_pid_link(df, register_sp_link, join_link_col, target_link_col)

        df = map_pids_to_ints(df, subject_id_mapping)

        df = clean_data(df)

        return df

    def _apply_sp_pid_link(
        df: pd.DataFrame,
        register_sp_link: pd.DataFrame,
        join_link_col: str,
        target_link_col: str,
    ) -> pd.DataFrame:
        """
        Apply SP PID link.
        We can expect the subject_id is present in df at the end of processing.
        The column names in the link file will be provided via config.
        There will be a join column and a target column and we can essentially reuse our apply_mapping function,
        just accessing args differently.
        """
        if SUBJECT_ID not in df.columns:
            raise ValueError(f"SUBJECT_ID column not found in df: {df.columns}")
        return apply_mapping(
            df,
            register_sp_link,
            join_col=join_link_col,
            source_col=SUBJECT_ID,
            target_col=target_link_col,
            how="inner",
            rename_to=SUBJECT_ID,
            drop_source=True,
        )

    @staticmethod
    def _apply_mappings(df: pd.DataFrame, concept_config: dict, data_handler: "DataHandler") -> pd.DataFrame:
        if concept_config.get("mappings"):
            for mapping in concept_config.mappings:
                map_table = data_handler.load_pandas(
                    mapping["via_file"],
                    cols=[mapping["join_on"], mapping["target_column"]],
                )
                df = apply_mapping(
                    df,
                    map_table,
                    join_col=mapping["join_on"],
                    source_col=mapping["source_column"],
                    target_col=mapping["target_column"],
                    rename_to=mapping["rename_to"],
                    how=mapping.get("how", "inner"),
                    drop_source=mapping.get("drop_source", False),
                )
        return df

    @staticmethod
    def _unroll_columns(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
        """Unroll columns if needed."""
        if "unroll_columns" in concept_config:
            processed_dfs = unroll_columns(df, concept_config)
            return pd.concat(processed_dfs, ignore_index=True) if processed_dfs else df
        return df

    @staticmethod
    def _combine_datetime_columns(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
        """Combine date and time columns into datetime columns."""
        if "combine_datetime" in concept_config:
            for target_col, date_time_cols in concept_config["combine_datetime"].items():
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
