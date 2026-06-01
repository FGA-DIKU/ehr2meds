import pandas as pd
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.utils import (
    apply_mapping,
    apply_value_map,
    clean_data,
    convert_numeric_columns,
    convert_timestamp_columns,
    fill_missing_values,
    map_pids_to_ints,
    melt_table,
    normalize_columns,
    pad_values,
    prefix_codes,
    select_and_rename_columns,
    unroll_columns,
)
from pathlib import Path
from typing import Dict, Optional


class Processor:
    @staticmethod
    def process(
        df: pd.DataFrame,
        table_config: dict,
        subject_id_mapping: Dict[str, int],
        data_handler: "DataHandler",
        time_stamp_dict: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Process the table.
        1. Normalize columns
        2. Apply value mappings
        3. Select and rename columns
        4. apply columns map
        5. Pad values
        6. fill missing values
        7. combine datetime columns
        8. unroll columns (process codes)
        9. convert numeric columns
        10. apply pid integer mapping
        11. clean data
        """
        df = normalize_columns(df, table_config)
        df = apply_value_map(df, table_config)
        df = select_and_rename_columns(df, table_config.get("rename_columns", {}))
        df = Processor._apply_mappings(df, table_config, data_handler)
        df = pad_values(df, table_config)
        df = fill_missing_values(df, table_config.get("fillna", {}))
        df = melt_table(df, table_config.get("melt_table", {}))
        df = Processor._combine_datetime_columns(df, table_config)
        df = Processor._combine_datetime_from_parts(df, table_config)
        df = prefix_codes(df, table_config.get("code_prefix", None))
        df = Processor._unroll_columns(df, table_config)
        df = convert_numeric_columns(df, table_config)
        if time_stamp_dict:
            df = convert_timestamp_columns(df, **time_stamp_dict)
        df = map_pids_to_ints(df, subject_id_mapping)
        df = clean_data(df)

        return df

    @staticmethod
    def _get_mapping_table(data_handler, mapping):
        "Find path and relevant columns in either registry or the resources folder."
        filename = mapping["via_file"]
        cols = [mapping["join_on"], mapping["target_column"]]

        register_path = Path(filename)
        if not register_path.exists():
            filename = str(Path(__file__).parent.parent / "resources" / filename)

        return data_handler.load_pandas(filename, cols=cols)

    @staticmethod
    def _apply_mappings(df: pd.DataFrame, table_config: dict, data_handler: "DataHandler") -> pd.DataFrame:
        if table_config.get("mappings"):
            for mapping in table_config.mappings:
                map_table = Processor._get_mapping_table(data_handler, mapping)
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
    def _unroll_columns(df: pd.DataFrame, table_config: dict) -> pd.DataFrame:
        """Unroll columns if needed."""
        if "unroll_columns" in table_config:
            processed_dfs = unroll_columns(df, table_config)
            return pd.concat(processed_dfs, ignore_index=True) if processed_dfs else df
        return df

    @staticmethod
    def _combine_datetime_columns(df: pd.DataFrame, table_config: dict) -> pd.DataFrame:
        """Combine date and time columns into datetime columns."""
        if "combine_datetime" in table_config:
            for target_col, date_time_cols in table_config["combine_datetime"].items():
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
    def _combine_datetime_from_parts(df: pd.DataFrame, table_config: dict) -> pd.DataFrame:
        """Combine date and time columns into datetime columns."""
        if "combine_datetime_parts" in table_config:
            for target_col, date_time_cols in table_config["combine_datetime_parts"].items():
                date_col = date_time_cols.get("date_col")
                hour_col = date_time_cols.get("hour_col")
                minute_col = date_time_cols.get("minute_col")

                if date_col in df.columns:
                    dt_str = (
                        df[date_col].dt.strftime("%Y-%m-%d")
                        if pd.api.types.is_datetime64_any_dtype(df[date_col])
                        else df[date_col].astype(str)
                    )

                    hour = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
                    minute = pd.to_numeric(df[minute_col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(2)
                    dt_str = dt_str + " " + hour + ":" + minute + ":00"
                    df[target_col] = pd.to_datetime(dt_str, errors="coerce")
                    # Drop original columns if requested
                    if date_time_cols.get("drop_original", True):
                        df = df.drop(columns=[date_col, hour_col, minute_col])
        return df
