import pandas as pd
import yaml
import inspect
import argparse
from pathlib import Path
import ehr2meds.data.generation.helpers as ghelpers
import ehr2meds.data.corruption.helpers as chelpers
import random

if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets based on a YAML configuration."
    )
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()


    root = Path("ehr2meds")

    with open(root / "data" / "generation" / args.config) as f:
        cfg = yaml.safe_load(f)
    output_dir = root / cfg["paths"]["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    gfunc_dict = {
        name: obj for name, obj in inspect.getmembers(ghelpers) if inspect.isfunction(obj)
    }
    cfunc_dict = {
        name: obj for name, obj in inspect.getmembers(chelpers) if inspect.isfunction(obj)
    }

    # Iterate through each file and its corresponding configuration
    for file, info in cfg["data"].items():
        df = []
        for i in range(info["N"]):
            row = {}

            for col, col_info in info["columns"].items():
                if col_info["type"] not in gfunc_dict:
                    raise ValueError(f"Unknown generation function type: {col_info['type']}")
                func = gfunc_dict[col_info["type"]]

                call_args = col_info.get("args", {})
                # Handle dependencies between columns using the "match" key
                if "match" in col_info:
                    keyword, match_col = next(iter(col_info["match"].items()))
                    call_args[keyword] = row[match_col]
                value = func(**call_args)

                # Apply column-specific corruptions if specified
                if "corruptions" in col_info:
                    for corruption in col_info["corruptions"]:
                        if corruption["type"] not in cfunc_dict:
                            raise ValueError(f"Unknown corruption function type: {corruption['type']}")
                        cfunc = cfunc_dict[corruption["type"]]
                        value = cfunc(value, row_index=i, **corruption.get("args", {}))

                row[col] = value

            for corruption in info.get("corruptions", []):
                row = (
                    row.copy()
                )  # Avoid modifying the original row for subsequent corruptions
                if corruption["type"] not in cfunc_dict:
                    raise ValueError(f"Unknown corruption function type: {corruption['type']}")
                func = cfunc_dict[corruption["type"]]
                row = func(row, row_index=i, **corruption.get("args", {}))
            df.append(row)

        df = pd.DataFrame(df)
        df.to_csv(output_dir / f"{file}.csv", index=False)
