import pandas as pd
import yaml
import inspect
import argparse
from pathlib import Path
import ehr2meds.data.summary.filter_helpers as filter_helpers
import ehr2meds.data.summary.extract_helpers as extract_helpers
import random
import os 


class TableBuilder:
    def __init__(self, filter_func_dict, extract_func_dict, main_cfg, input_path):
        self.filter_func_dict = filter_func_dict
        self.extract_func_dict = extract_func_dict
        main_df = self.get_main_df(main_cfg)
        print(main_df.head())

    def get_main_df(self, cfg, input_path):
        main_df = pd.DataFrame()
        for source in cfg["sources"]:
            source_path = os.path.join(input_path, source["source_file"])
            source_df = pd.read_csv(source_path)
            main_df = main_df.merge(source_df, on=cfg["merge_columns"], how="left")
        
        for column in cfg["add_columns"]:
            fn = self.extract_func_dict[column["function"]]
            main_df[column] = fn(main_df, **column["args"])
            
        return main_df

    # def run(self, cfg, output_dir):
    #     input_path = cfg["paths"]["input"]

    #     tables = []

    #     for linked_name, linked_cfg in cfg["summary"]["linked_columns"].items():
    #         source_path = os.path.join(input_path, linked_cfg["source_file"])
    #         linked_df = self.get_linked_df(linked_cfg, source_path)

    #         tabl_type = linked_cfg["type"]               
    #         tabl_func = tabl_type["function"]           
    #         fn = self.extract_func_dict[tabl_func]            
    #         args_dict = tabl_type.get("args", {})      
    #         linked_table = fn(linked_df, linked_name, self.required_columns, **args_dict)
    #         tables.append(linked_table)

    #     final_table = self.merge_tables(tables)
    #     print(len(final_table))
    #     final_table.to_csv(output_dir / "final_table.csv", index=False)
    #     print(final_table.head())
    #     print("Saved final table to", output_dir / "final_table.csv")

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

    table_builder = TableBuilder(filter_func_dict, extract_func_dict, cfg["main_table"], input_dir)
    # table_builder.get_main_df(cfg["main_table"])
