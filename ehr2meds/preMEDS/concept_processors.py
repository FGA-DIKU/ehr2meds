import pandas as pd
from ehr2meds.preMEDS.constants import SUBJECT_ID
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.utils import (
    apply_mapping,
    clean_data,
    convert_numeric_columns,
    convert_timestamp_columns,
    fill_missing_values,
    map_pids_to_ints,
    melt_table,
    select_and_rename_columns,
    unroll_columns,
    normalize_columns,
    apply_value_map,
    replace_values,
    pad_values
)
from typing import Dict, Optional


class SPConceptProcessor:
    @staticmethod
    def process(
        df: pd.DataFrame,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        time_stamp_dict: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Main method for processing a single concept's data
        """
        df = select_and_rename_columns(df, concept_config.get("rename_columns", {}))
        if concept_config.get("fillna"):
            df = fill_missing_values(df, concept_config.fillna)

        if concept_config.get("melt_table"):
            df = melt_table(df, concept_config)

        if time_stamp_dict:
            df = convert_timestamp_columns(df, **time_stamp_dict)

        df = convert_numeric_columns(df, concept_config)
        df = map_pids_to_ints(df, subject_id_mapping)
        df = clean_data(df)

        return df


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
        time_stamp_dict: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Process the register concepts.
        1. Replace values
        2. Normalize columns
        3. Apply value mappings
        4. Select and rename columns
        5. apply columns map
        6. Pad values
        7. fill missing values
        8. combine datetime columns
        9. unroll columns (process codes)
        10. convert numeric columns
        11. apply pid linking
        12. apply pid integer mapping
        13. clean data
        """
        df = replace_values(df, concept_config)
        df = normalize_columns(df, concept_config)
        df = apply_value_map(df, concept_config)
        df = select_and_rename_columns(df, concept_config.get("rename_columns", {}))
        df = RegisterConceptProcessor._apply_mappings(df, concept_config, data_handler)
        df = pad_values(df, concept_config)
        df = fill_missing_values(df, concept_config.get("fillna", {}))
        df = RegisterConceptProcessor._combine_datetime_columns(df, concept_config)
        df = RegisterConceptProcessor._combine_datetime_from_parts(df, concept_config)

        if time_stamp_dict:
            df = convert_timestamp_columns(df, **time_stamp_dict)

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


    @staticmethod
    def _combine_datetime_from_parts(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
        """Combine date and time columns into datetime columns."""
        if "combine_datetime_parts" in concept_config:
            for target_col, date_time_cols in concept_config["combine_datetime_parts"].items():
                date_col = date_time_cols.get("date_col")
                hour_col = date_time_cols.get("hour_col")
                minute_col = date_time_cols.get("minute_col")

                if date_col in df.columns:
                    dt_str = df[date_col].dt.strftime("%Y-%m-%d") if pd.api.types.is_datetime64_any_dtype(df[date_col]) else df[date_col].astype(str)
                    # dt_str = df[date_col].astype(str)
                    hour = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
                    minute = pd.to_numeric(df[minute_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
                    dt_str = dt_str + " " + hour + ":" + minute + ":00"
                    df[target_col] = pd.to_datetime(dt_str, errors = "coerce")
                    # Drop original columns if requested
                    if date_time_cols.get("drop_original", True):
                        df = df.drop(columns=[date_col, hour_col, minute_col])
        return df
