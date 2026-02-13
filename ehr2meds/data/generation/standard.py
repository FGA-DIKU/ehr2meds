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

    def generate_rows(self, info, row, row_index):
        for col, col_info in info["columns"].items():
            if col_info["type"] not in self.gfunc_dict:
                raise ValueError(
                    f"Unknown generation function type: {col_info['type']}"
                )
            func = self.gfunc_dict[col_info["type"]]

            call_args = col_info.get("args", {})
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
                df.append(row)

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
    standard_generator = StandardGenerator(gfunc_dict, cfunc_dict)
    standard_generator.generate_data_files(cfg, output_dir)