from typing import Dict

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import CODE, SUBJECT_ID
from ehr2meds.PREMEDS.preprocessing.io.data_handling import (
    DataHandler,
    load_mapping_file,
)
from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import (
    apply_secondary_mapping,
    convert_numeric_columns,
    map_and_clean_data,
    select_and_rename_columns,
    unroll_columns,
)


class RegisterConceptProcessor:
    @staticmethod
    def process_register_concept(
        df: pd.DataFrame,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        data_handler: "DataHandler",
        register_sp_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process the register concepts.
        1. Select and rename columns
        
        
        """
        # Step 1: Select columns
        df = select_and_rename_columns(df, concept_config.get("rename_columns", {}))

        # Combine datetime columns if needed
        df = RegisterConceptProcessor._combine_datetime_columns(df, concept_config)

        # Step 2: Apply secondary mapping if needed
        df = apply_secondary_mapping(df, concept_config, data_handler)

        # Step 3: Convert numeric columns
        df = convert_numeric_columns(df, concept_config)

        # Step 4: Apply main mapping and register mapping
        df = RegisterConceptProcessor._apply_main_and_register_mapping(
            df, concept_config, data_handler, register_sp_mapping
        )

        # Step 5: Process codes (unroll or prefix)
        df = RegisterConceptProcessor._process_register_codes(df, concept_config)

        # Step 6: Final cleanup and mapping
        df = map_and_clean_data(df, subject_id_mapping)

        return df
        #

    @staticmethod
    def _process_initial_register_data(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Handle initial data processing steps."""
        # Select and rename columns


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
        mapping_df = load_mapping_file(mapping_cfg, data_handler)

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
            processed_dfs = unroll_columns(df, concept_config)
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
