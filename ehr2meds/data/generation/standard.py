import pandas as pd
import yaml
import inspect
import argparse
import helpers
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate synthetic datasets based on a YAML configuration.")
parser.add_argument("--config", type=str, help="Path to the YAML configuration file.")
args = parser.parse_args()


root = Path("ehr2meds")

with open(root / "data" / "generation" / args.config) as f:
    cfg = yaml.safe_load(f)
output_dir = (root / cfg["paths"]["output"])
output_dir.mkdir(parents=True, exist_ok=True)

func_dict = {name: obj for name, obj in inspect.getmembers(helpers) if inspect.isfunction(obj)}

for file, info in cfg["data"].items():
    df = []
    N = info["N"]
    for i in range(N):
        row = {}
        for col, col_info in info["columns"].items():
            func = func_dict[col_info["type"]]
            if "match" in col_info:
                keyword, match_col = next(iter(col_info["match"].items()))
                col_info.setdefault("args", {})[keyword] = row[match_col]
            row[col] = func(**col_info.get("args", {}))
        df.append(row)
    
    df = pd.DataFrame(df)
    df.to_csv(output_dir / f"{file}.csv", index=False)