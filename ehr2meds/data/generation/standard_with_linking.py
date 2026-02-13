import pandas as pd
import yaml
import inspect
import argparse
from pathlib import Path
import random
from standard import StandardGenerator
import ehr2meds.data.generation.helpers as ghelpers
import ehr2meds.data.corruption.helpers as chelpers


class StandardWithLinkingGenerator(StandardGenerator):
    def __init__(self, gfunc_dict, cfunc_dict):
        super().__init__(gfunc_dict, cfunc_dict)
        self.gfunc_dict = gfunc_dict
        self.cfunc_dict = cfunc_dict

    def generate_linked_columns(self, info, row):
        for col, col_info in info["linked_columns"].items():
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
            final_col_name = rename_to if rename_to else col
            row[final_col_name] = selected_row
        return row

    def generate_linked_data_files(self, cfg, output_dir):
        # 1: Generate main data files using the standard generation function
        if "data" in cfg:
            self.generate_data_files(cfg, output_dir)

        # 2: Generate linked data files
        for file, info in cfg.get("linked_data", {}).items():
            df = []
            for i in range(info["N"]):
                row = {}
                row = self.generate_rows(info, row, i)
                row = self.generate_linked_columns(info, row)
                row = self.generate_corruptions(info, row, i)
                df.append(row)

            df = pd.DataFrame(df)
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
    standard_with_linking_generator = StandardWithLinkingGenerator(gfunc_dict, cfunc_dict)
    standard_with_linking_generator.generate_linked_data_files(cfg, output_dir)