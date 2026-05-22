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
    pad_values,
)
from typing import Dict, Optional
from pathlib import Path


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
        11. apply pid integer mapping
        12. clean data
        """
        label = concept_config.get("filename", "register")
        print(f"[DEBUG {label}] start rows={len(df)} cols={list(df.columns)}", flush=True)

        print(f"[DEBUG {label}] step 1: replace_values", flush=True)
        df = replace_values(df, concept_config)
        print(f"[DEBUG {label}] step 2: normalize_columns", flush=True)
        df = normalize_columns(df, concept_config)
        print(f"[DEBUG {label}] step 3: apply_value_map", flush=True)
        df = apply_value_map(df, concept_config)
        print(f"[DEBUG {label}] step 4: select_and_rename_columns", flush=True)
        df = select_and_rename_columns(df, concept_config.get("rename_columns", {}))
        print(f"[DEBUG {label}] step 5: apply_mappings rows={len(df)} cols={list(df.columns)}", flush=True)
        df = RegisterConceptProcessor._apply_mappings(df, concept_config, data_handler)
        print(f"[DEBUG {label}] step 6: pad_values", flush=True)
        df = pad_values(df, concept_config)
        print(f"[DEBUG {label}] step 7: fill_missing_values", flush=True)
        df = fill_missing_values(df, concept_config.get("fillna", {}))
        print(f"[DEBUG {label}] step 8: combine_datetime_columns", flush=True)
        df = RegisterConceptProcessor._combine_datetime_columns(df, concept_config)
        print(f"[DEBUG {label}] step 9: combine_datetime_from_parts cols={list(df.columns)}", flush=True)
        df = RegisterConceptProcessor._combine_datetime_from_parts(df, concept_config)
        print(f"[DEBUG {label}] after datetime parts cols={list(df.columns)}", flush=True)

        if time_stamp_dict:
            print(f"[DEBUG {label}] step 10: convert_timestamp_columns", flush=True)
            df = convert_timestamp_columns(df, **time_stamp_dict)

        print(f"[DEBUG {label}] step 11: unroll_columns", flush=True)
        df = RegisterConceptProcessor._unroll_columns(df, concept_config)
        print(f"[DEBUG {label}] after unroll rows={len(df)} cols={list(df.columns)}", flush=True)

        print(f"[DEBUG {label}] step 12: convert_numeric_columns", flush=True)
        df = convert_numeric_columns(df, concept_config)

        if SUBJECT_ID not in df.columns:
            raise ValueError(
                f"{SUBJECT_ID} column not found after processing: {list(df.columns)}. "
                "Set subject_id via rename_columns or mappings."
            )

        print(f"[DEBUG {label}] step 13: map_pids_to_ints", flush=True)
        df = map_pids_to_ints(df, subject_id_mapping)
        print(f"[DEBUG {label}] step 14: clean_data", flush=True)
        df = clean_data(df)
        print(f"[DEBUG {label}] done rows={len(df)}", flush=True)

        return df

    @staticmethod
    def _get_mapping_table(data_handler, mapping):
        "Find path and relevant columns in either registry or the resources folder."
        filename = mapping["via_file"]
        cols = [mapping["join_on"], mapping["target_column"]]

        register_path = Path(data_handler.data_loader.path) / filename
        if not register_path.exists():
            filename = str(Path(__file__).parent.parent / "resources" / filename)

        return data_handler.load_pandas(filename, cols=cols)

    @staticmethod
    def _apply_mappings(
        df: pd.DataFrame, concept_config: dict, data_handler: "DataHandler"
    ) -> pd.DataFrame:
        if concept_config.get("mappings"):
            for mapping in concept_config.mappings:
                map_table = RegisterConceptProcessor._get_mapping_table(
                    data_handler, mapping
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

    @staticmethod
    def _combine_datetime_from_parts(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Combine date and time columns into datetime columns."""
        if "combine_datetime_parts" in concept_config:
            for target_col, date_time_cols in concept_config[
                "combine_datetime_parts"
            ].items():
                date_col = date_time_cols.get("date_col")
                hour_col = date_time_cols.get("hour_col")
                minute_col = date_time_cols.get("minute_col")

                print(
                    f"[DEBUG combine_datetime_parts] target={target_col} "
                    f"date={date_col} hour={hour_col} minute={minute_col} "
                    f"present={{date: {date_col in df.columns}, "
                    f"hour: {hour_col in df.columns}, "
                    f"minute: {minute_col in df.columns}}}",
                    flush=True,
                )
                if date_col in df.columns:
                    dt_str = (
                        df[date_col].dt.strftime("%Y-%m-%d")
                        if pd.api.types.is_datetime64_any_dtype(df[date_col])
                        else df[date_col].astype(str)
                    )
                    # dt_str = df[date_col].astype(str)
                    hour = (
                        pd.to_numeric(df[hour_col], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .astype(str)
                        .str.zfill(2)
                    )
                    minute = (
                        pd.to_numeric(df[minute_col], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .astype(str)
                        .str.zfill(2)
                    )
                    dt_str = dt_str + " " + hour + ":" + minute + ":00"
                    df[target_col] = pd.to_datetime(dt_str, errors="coerce")
                    # Drop original columns if requested
                    if date_time_cols.get("drop_original", True):
                        df = df.drop(columns=[date_col, hour_col, minute_col])
        return df
