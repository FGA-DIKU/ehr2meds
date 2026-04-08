import pandas as pd
import yaml
import inspect
import argparse
from pathlib import Path
import ehr2meds.data.extraction.helpers.extract_helpers as extract_helpers
import ehr2meds.data.extraction.helpers.collapse_helpers as collapse_helpers
import ehr2meds.data.extraction.helpers.link_helpers as link_helpers
import random
import os 


class TableBuilder:
    def __init__(self, link_func_dict, extract_func_dict, collapse_func_dict, main_cfg, input_path):
        self.link_func_dict = link_func_dict
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

        rename_columns = cfg.get("rename_columns") or {}
        if rename_columns:
            main_df.rename(columns=rename_columns, inplace=True)

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
        link_func = self.link_func_dict[fn_name]
        args = dict(rule.get("args") or {})
        args.setdefault("name", rule["name"])

        return link_func(
            df=linked_df,
            match_on=match_on,
            expanded_table=expanded_table,
            **args,
        )
    
    def _check_new_rows(self, expanded_table: pd.DataFrame, res: pd.DataFrame, out_col: str):
        if len(res) != len(expanded_table) or not res.index.equals(expanded_table.index):
            raise ValueError(
                f"Linked rule {out_col!r} returned a table with different rows/index "
                f"(expected len={len(expanded_table)}, got len={len(res)})."
            )
        return res

    def get_expanded_table(self, cfg: dict, input_path):
        expanded_table = self.main_df.copy()

        linked_tables_cfg = cfg.get("linked_tables") or []
        for rule in linked_tables_cfg:
            out_col = rule["name"]
            res = self._apply_linked_rule(expanded_table, rule, input_path)
            self._check_new_rows(expanded_table, res, out_col)
            expanded_table[out_col] = res[out_col]
            print(expanded_table.head(20))
        return expanded_table

    def get_collapsed_table(self, expanded_table: pd.DataFrame, cfg: dict):
        final_table = self.main_df.copy()

        for collapse_spec in (cfg.get("collapse_table") or []):
            out_col = collapse_spec["name"]
            fn_name = collapse_spec["combine"]["function"]
            collapse_func = self.collapse_func_dict[fn_name]
            res = collapse_func(
                expanded_table=expanded_table,
                name=out_col,
                components=collapse_spec.get("components") or [],
            )
            self._check_new_rows(expanded_table, res, out_col)
            final_table[out_col] = res[out_col]
        return final_table

    def run(self, cfg: dict, input_path, output_dir, save_name):
        expanded_table = self.get_expanded_table(cfg, input_path)
        final_table = self.get_collapsed_table(expanded_table, cfg)

        print(final_table.head(20))
        final_table.to_csv(output_dir / f"{save_name}.csv", index=False)
        print("Saved final table to", output_dir / f"{save_name}.csv")

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
    parser.add_argument(
        "--save_name", type=str, help="Name of the table to save.",
        default="final_table",
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
    link_func_dict = {
        name: obj
        for name, obj in inspect.getmembers(link_helpers)
        if inspect.isfunction(obj)
    }
    collapse_func_dict = {
        name: obj
        for name, obj in inspect.getmembers(collapse_helpers)
        if inspect.isfunction(obj)
    }

    table_builder = TableBuilder(link_func_dict, extract_func_dict, collapse_func_dict, cfg["main_table"], input_dir)
    table_builder.run(cfg, input_dir, output_dir, args.save_name)
