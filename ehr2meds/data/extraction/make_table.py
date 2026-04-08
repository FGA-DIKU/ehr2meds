import pandas as pd
import yaml
import inspect
import argparse
from pathlib import Path
import ehr2meds.data.extraction.filter_helpers as filter_helpers
import ehr2meds.data.extraction.extract_helpers as extract_helpers
import ehr2meds.data.extraction.collapse_helpers as collapse_helpers
import random
import os 


class TableBuilder:
    def __init__(self, filter_func_dict, extract_func_dict, collapse_func_dict, main_cfg, input_path):
        self.filter_func_dict = filter_func_dict
        self.extract_func_dict = extract_func_dict
        self.collapse_func_dict = collapse_func_dict
        self.main_df = self.get_main_df(main_cfg, input_path)
        print(self.main_df.head())

    def get_main_df(self, cfg, input_path):
        files = list(cfg["sources"])
        if not files:
            main_df = pd.DataFrame()
        else:
            main_df = pd.read_csv(os.path.join(input_path, files[0]))
            for fname in files[1:]:
                other = pd.read_csv(os.path.join(input_path, fname))
                merged = main_df.merge(
                    other,
                    on=cfg["merge_columns"],
                    how="outer",
                    indicator=True,
                )
                n_left_only = int((merged["_merge"] == "left_only").sum())
                n_right_only = int((merged["_merge"] == "right_only").sum())
                n_both = int((merged["_merge"] == "both").sum())
                print(
                    f"Merging '{fname}' on {cfg['merge_columns']}: "
                    f"both={n_both}, left_only={n_left_only}, right_only={n_right_only} "
                    f"(total={len(merged)})"
                )
                main_df = merged.drop(columns=["_merge"])

        for spec in cfg.get("add_columns") or []:
            fn = self.extract_func_dict[spec["function"]]
            out_name = spec["name"]
            args_spec = spec.get("args") or {}
            kwargs = {param: main_df[col] for param, col in args_spec.items()}
            main_df[out_name] = fn(**kwargs)

        return main_df

    def _apply_linked_rule(self, expanded_table: pd.DataFrame, rule: dict, input_path) -> pd.DataFrame:
        """
        Apply a single linked rule and return a boolean Series aligned to expanded_table.

        Expected rule schema (list style):
          - name: <output column name>
          - source_file: <csv>
          - match_on: [..] OR key_map: {main_col: linked_col}
          - function: <collapse_helpers function name> (default: bool_in_time_window)
          - args: {...} passed to the collapse function
        """
        source_file = rule["source_file"]
        source_path = os.path.join(input_path, source_file)
        linked_df = pd.read_csv(source_path)

        key_map = rule.get("key_map")
        if key_map:
            linked_df = linked_df.rename(columns=dict(key_map))
            match_on = list(key_map.keys())
        else:
            match_on = list(rule.get("match_on") or [])

        fn_name = rule.get("function")
        link_func = self.collapse_func_dict[fn_name]
        args = dict(rule.get("args") or {})
        args.setdefault("name", rule["name"])

        return link_func(
            df=linked_df,
            match_on=match_on,
            expanded_table=expanded_table,
            **args,
        )

    def run(self, linked_tables_cfg: list[dict], input_path, output_dir):
        expanded_table = self.main_df.copy()

        if linked_tables_cfg is None:
            linked_tables_cfg = []

        for rule in linked_tables_cfg:
            out_col = rule["name"]
            res = self._apply_linked_rule(expanded_table, rule, input_path)
            expanded_table = res
            print(expanded_table.head())

        print(expanded_table.head())
        # main_table = self.main_df
        # for collapse_table in cfg["collapse_tables"]:
        #     collapse_func = self.collapse_func_dict[collapse_table["combine"]["function"]]
        #     sub_table = collapse_func(expanded_table, collapse_table["components"])
        #     expanded_table = expanded_table.merge(sub_table, on=collapse_table["match_on"], how="left")
        
        # for linked_name, linked_cfg in cfg["summary"]["linked_columns"].items():
        #     source_path = os.path.join(input_path, linked_cfg["source_file"])
        #     linked_df = self.get_linked_df(linked_cfg, source_path)

        #     tabl_type = linked_cfg["type"]               
        #     tabl_func = tabl_type["function"]           
        #     fn = self.extract_func_dict[tabl_func]            
        #     args_dict = tabl_type.get("args", {})      
        #     linked_table = fn(linked_df, linked_name, self.required_columns, **args_dict)
        #     tables.append(linked_table)

        # final_table = self.merge_tables(tables)
        # print(len(final_table))
        # final_table.to_csv(output_dir / "final_table.csv", index=False)
        # print(final_table.head())
        # print("Saved final table to", output_dir / "final_table.csv")

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
    collapse_func_dict = {
        name: obj
        for name, obj in inspect.getmembers(collapse_helpers)
        if inspect.isfunction(obj)
    }

    table_builder = TableBuilder(filter_func_dict, extract_func_dict, collapse_func_dict, cfg["main_table"], input_dir)
    table_builder.run(cfg.get("linked_tables") or [], input_dir, output_dir)
