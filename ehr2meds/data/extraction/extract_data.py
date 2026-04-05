import argparse
import inspect
import random
from pathlib import Path

import pandas as pd
import yaml

import ehr2meds.data.extraction.extract_helpers as extract_helpers


class DataExtractor:
    def __init__(self, extract_func_dict, input_dir: Path):
        self.extract_func_dict = extract_func_dict
        self.input_dir = Path(input_dir)

    def run(self, cfg, output_dir: Path):
        output_dir = Path(output_dir)
        for data_name, data_cfg in cfg["data_extraction"].items():
            res_df = pd.DataFrame(columns=data_cfg["key_columns"])
            for source in data_cfg["sources"]:
                path = self.input_dir / source["source_file"]
                source_df = pd.read_csv(path)
                fn = source["type"]["function"]
                source_df = self.extract_func_dict[fn](
                    source_df, **source["type"]["args"]
                )
                map_columns = source["map_columns"]
                source_df = source_df[list(map_columns.keys())].rename(
                    columns=map_columns
                )
                if res_df.empty:
                    res_df = source_df
                else:
                    res_df = res_df.merge(source_df, on=source["key_columns"], how="left")
            res_df.to_csv(output_dir / f"{data_name}.csv", index=False)

if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser(
        description="Extract summary tables from raw CSVs using a YAML config."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing source CSV files referenced in the config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to write extracted CSV tables.",
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
        if inspect.isfunction(obj) and obj.__module__ == extract_helpers.__name__
    }

    data_extractor = DataExtractor(extract_func_dict, input_dir)
    data_extractor.run(cfg, output_dir)
