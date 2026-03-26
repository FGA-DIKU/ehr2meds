import pandas as pd
import yaml
import inspect
import argparse
from pathlib import Path
import ehr2meds.data.summary.filter_helpers as filter_helpers
import ehr2meds.data.summary.extract_helpers as extract_helpers
import random
import os 

def verify_main_df(main_df: pd.DataFrame, input_table_cfg: dict) -> tuple[pd.DataFrame, list[str], str | None, str | None]:
    """
    Validate that the primary dataframe contains:
    - all key columns
    - all additional columns (if any)
    - time window endpoint columns referenced by input_table_cfg["time_window"] (min_date/max_date, if present)
    """
    key_columns = input_table_cfg["key_columns"]
    additional_columns = input_table_cfg.get("additional_columns") or []
    time_window = input_table_cfg.get("time_window", {}) or {}

    min_date_col = time_window.get("min_date")
    max_date_col = time_window.get("max_date")

    required_cols: list[str] = list(key_columns) + list(additional_columns)
    if min_date_col:
        required_cols.append(min_date_col)
    if max_date_col:
        required_cols.append(max_date_col)

    required_cols = list(set(required_cols))

    missing = [c for c in required_cols if c not in main_df.columns]
    if missing:
        raise ValueError(
            f"Primary table {input_table_cfg['path']} is missing required columns: {missing}. "
            f"Available columns: {list(main_df.columns)}"
        )

    subset = main_df[required_cols].copy()
    if min_date_col and min_date_col in subset.columns:
        subset[min_date_col] = pd.to_datetime(subset[min_date_col], errors="coerce")
    if max_date_col and max_date_col in subset.columns:
        subset[max_date_col] = pd.to_datetime(subset[max_date_col], errors="coerce")

    return subset, list(key_columns), min_date_col, max_date_col, required_cols

class TableBuilder:
    def __init__(self, filter_func_dict, extract_func_dict, input_table_cfg):
        self.filter_func_dict = filter_func_dict
        self.extract_func_dict = extract_func_dict
        main_df = pd.read_csv(input_table_cfg["path"])
        print(len(main_df))

        self.main_df, self.key_columns, self.min_date, self.max_date, self.required_columns = verify_main_df(
            main_df, input_table_cfg
        )
        print(len(self.main_df))
        print(self.required_columns)

    # def filter_df(self, df, timestamp_col, filter_func):
    #     # Filter on time window
    #     if self.time_window["match"] == "max_date":
    #         df = df[df[timestamp_col] <= self.time_window["max_date"]]
    #     if self.time_window["match"] == "min_date":
    #         df = df[df[timestamp_col] >= self.time_window["min_date"]]

    #     # TODO optional extra filter function
    #     return df

    def get_linked_df(self, linked_cfg, source_path):
        linked_df = pd.read_csv(source_path)
        for (key, match) in linked_cfg["key_values"].items():
            linked_df.rename(columns={match: key}, inplace=True)
        missing_keys = [k for k in self.key_columns if k not in linked_df.columns]
        if missing_keys:
            raise ValueError(
                f"Linked file {source_path} is missing primary key columns after renaming: "
                f"{missing_keys}. Available columns: {list(linked_df.columns)}"
            )

        # Merge on key columns
        merged_df = pd.merge(self.main_df, linked_df, on=self.key_columns, how="left")
        linked_timestamp = linked_cfg["time_window"]["timestamp"]

        merged_df[linked_timestamp] = pd.to_datetime(merged_df[linked_timestamp], errors="coerce")

        # Apply per-row min/max window from primary columns.
        if self.min_date:
            merged_df = merged_df[merged_df[linked_timestamp] >= merged_df[self.min_date]]
        if self.max_date:
            merged_df = merged_df[merged_df[linked_timestamp] <= merged_df[self.max_date]]

        # TODO filter on function if provided
        if linked_cfg.get("filter"):
            filter_func = self.filter_func_dict[linked_cfg["filter"]["function"]]
            args = linked_cfg["filter"]["args"]
            merged_df = filter_func(merged_df, **args)
            print(merged_df.head())
        return merged_df

    def merge_tables(self, tables):
        final_table = self.main_df.copy()

        for table in tables:
            dup_mask = table.duplicated(subset=self.required_columns, keep=False)
            if dup_mask.any():
                dup_groups = table.loc[dup_mask, self.required_columns].drop_duplicates()
                sample = dup_groups.head(10).to_dict(orient="records")
                raise ValueError(
                    "Linked table contains duplicate rows for merge keys "
                    f"{self.required_columns}. Duplicate groups: {len(dup_groups)}. "
                    f"Sample: {sample}"
                )

            table_small = table
            final_table = final_table.merge(
                table_small,
                on=self.required_columns,
                how="left",
            )

        return final_table

    def run(self, cfg, output_dir):
        input_path = cfg["paths"]["input"]

        tables = []

        for linked_name, linked_cfg in cfg["summary"]["linked_columns"].items():
            source_path = os.path.join(input_path, linked_cfg["source_file"])
            linked_df = self.get_linked_df(linked_cfg, source_path)

            tabl_type = linked_cfg["type"]               
            tabl_func = tabl_type["function"]           
            fn = self.extract_func_dict[tabl_func]            
            args_dict = tabl_type.get("args", {})      
            linked_table = fn(linked_df, linked_name, self.required_columns, **args_dict)
            tables.append(linked_table)

        final_table = self.merge_tables(tables)
        print(len(final_table))
        final_table.to_csv(output_dir / "final_table.csv", index=False)
        print(final_table.head())
        print("Saved final table to", output_dir / "final_table.csv")

if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets based on a YAML configuration."
    )
    parser.add_argument(
        "--config", type=str, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--input", type=str, help="Path to the input directory."
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output directory."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_func_dict = {
        name: obj
        for name, obj in inspect.getmembers(extract_helpers)
        if inspect.isfunction(obj)
    }
    filter_func_dict = {
        name: obj
        for name, obj in inspect.getmembers(filter_helpers)
        if inspect.isfunction(obj)
    }

    table_builder = TableBuilder(filter_func_dict, extract_func_dict, cfg["summary"]["input_table"])
    table_builder.run(cfg, output_dir)
