import pandas as pd
import yaml
import inspect
import argparse
from pathlib import Path
import ehr2meds.data.generation.helpers as ghelpers
import ehr2meds.data.corruption.helpers as chelpers
import random


class StandardGenerator:
    def __init__(self, gfunc_dict, cfunc_dict):
        self.gfunc_dict = gfunc_dict
        self.cfunc_dict = cfunc_dict

    def handle_mix_function(self, call_args):
        callable_args = call_args.copy()
        for func_cfg in callable_args["functions"]:
            if func_cfg["name"] in self.gfunc_dict:
                func_cfg["func"] = self.gfunc_dict[func_cfg["name"]]
        return callable_args

    def generate_rows(self, info, row, row_index):
        for col, col_info in info["columns"].items():
            if col_info["type"] not in self.gfunc_dict:
                raise ValueError(
                    f"Unknown generation function type: {col_info['type']}"
                )
            func = self.gfunc_dict[col_info["type"]]

            call_args = col_info.get("args", {}).copy()
            if col_info["type"] == "mix_function":
                call_args = self.handle_mix_function(call_args)

            # Handle dependencies between columns using the "match" key
            if "match" in col_info:
                keyword, match_col = next(iter(col_info["match"].items()))
                call_args[keyword] = row[match_col]
            value = func(**call_args)

            # Apply column-specific corruptions if specified
            if "corruptions" in col_info:
                for corruption in col_info["corruptions"]:
                    if corruption["type"] not in self.cfunc_dict:
                        raise ValueError(
                            f"Unknown corruption function type: {corruption['type']}"
                        )
                    cfunc = self.cfunc_dict[corruption["type"]]
                    value = cfunc(value, row_index=row_index, **corruption.get("args", {}))

            row[col] = value
        return row

    def generate_corruptions(self, info, row, row_index):
        for corruption in info.get("corruptions", []):
            row = (
                row.copy()
            )  # Avoid modifying the original row for subsequent corruptions
            if corruption["type"] not in self.cfunc_dict:
                raise ValueError(
                    f"Unknown corruption function type: {corruption['type']}"
                )
            func = self.cfunc_dict[corruption["type"]]
            row = func(row, row_index=row_index, **corruption.get("args", {}))
        return row


    def generate_data_files(self, cfg, output_dir):
        # Iterate through each file and its corresponding configuration
        for file, info in cfg["data"].items():
            rows = []
            for i in range(info["N"]):
                row = {}
                row = self.generate_rows(info, row, i)
                row = self.generate_corruptions(info, row, i)
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_dir / f"{file}.csv", index=False)

class StandardWithLinkingGenerator(StandardGenerator):
    def __init__(self, gfunc_dict, cfunc_dict):
        super().__init__(gfunc_dict, cfunc_dict)

    def generate_linked_columns(self, info, row, output_dir):
        for _, col_info in info["linked_columns"].items():
            linked_file = col_info["file"]
            linked_on = col_info["linked_on"]
            rename_to = col_info.get("rename_to")
            linked_type = col_info["type"]

            # Load the linked DataFrame from CSV file
            linked_file_path = output_dir / f"{linked_file}.csv"
            if not linked_file_path.exists():
                raise ValueError(
                    f"Linked file '{linked_file_path}' not found. "
                    f"Make sure it's generated in the 'data' section first."
                )
            linked_df = pd.read_csv(linked_file_path)

            # Check the linked columns exist in linked file
            missing_linked_cols = [col for col in linked_on if col not in linked_df.columns]
            if missing_linked_cols:
                raise ValueError(
                    f"Linked columns {missing_linked_cols} not found in linked file '{linked_file}'. "
                    f"Available columns: {list(linked_df.columns)}"
                )

            # Get columns 
            linked_cols = linked_df[linked_on]

            if linked_type == "choice":
                selected_row = linked_cols.sample(n=1)
            else:
                raise ValueError(
                    f"Unknown linked type: {linked_type}"
                )
            # Insert column to row 
            if rename_to:
                selected_row.rename(columns=dict(zip(linked_on, rename_to)), inplace=True)

            row.update(selected_row.iloc[0])

        return row

    def generate_linked_data_files(self, cfg, output_dir):
        # 1: Generate main data files using the standard generation function
        if "data" in cfg:
            self.generate_data_files(cfg, output_dir)

        # 2: Generate linked data files
        for file, info in cfg.get("linked_data", {}).items():
            rows = []
            for i in range(info["N"]):
                row = {}
                row = self.generate_rows(info, row, i)
                row = self.generate_linked_columns(info, row, output_dir)
                row = self.generate_corruptions(info, row, i)
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_dir / f"{file}.csv", index=False)

if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets based on a YAML configuration."
    )
    parser.add_argument(
        "--config", type=str, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    root = Path("ehr2meds")

    with open(root / "data" / "generation" / args.config) as f:
        cfg = yaml.safe_load(f)
    output_dir = root / cfg["paths"]["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    gfunc_dict = {
        name: obj
        for name, obj in inspect.getmembers(ghelpers)
        if inspect.isfunction(obj)
    }
    cfunc_dict = {
        name: obj
        for name, obj in inspect.getmembers(chelpers)
        if inspect.isfunction(obj)
    }

    if "linked_data" in cfg:
        standard_generator = StandardWithLinkingGenerator(gfunc_dict, cfunc_dict)
        standard_generator.generate_linked_data_files(cfg, output_dir)
    else:
        standard_generator = StandardGenerator(gfunc_dict, cfunc_dict)
        standard_generator.generate_data_files(cfg, output_dir)