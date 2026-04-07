import pandas as pd
import yaml
import inspect
import argparse
from pathlib import Path
import ehr2meds.synthetic_data_generation.generate as generators
import ehr2meds.synthetic_data_generation.corrupt as corruptors
import random


def handle_mix_function(call_args, generators_dict):
    callable_args = call_args.copy()
    for func_cfg in callable_args["functions"]:
        if func_cfg["type"] in generators_dict:
            func_cfg["func"] = generators_dict[func_cfg["type"]]
        else:
            raise ValueError(
                f"Function {func_cfg['type']} not found in generators_dict"
            )
    return callable_args


def generate_rows(table_cfg, row, row_index, generators_dict, corruptors_dict):
    for column_name, col_cfg in table_cfg["columns"].items():
        if col_cfg["type"] not in generators_dict:
            raise ValueError(f"Unknown generation function type: {col_cfg['type']}")
        func = generators_dict[col_cfg["type"]]

        col_args = col_cfg.get("args", {}).copy()
        if col_cfg["type"] == "mix_function":
            col_args = handle_mix_function(col_args, generators_dict)

        # Handle dependencies between columns using the "match" key
        if "match" in col_cfg:
            keyword, match_col = next(iter(col_cfg["match"].items()))
            col_args[keyword] = row[match_col]
        value = func(**col_args)

        # Apply column-specific corruptions if specified
        if "corruptions" in col_cfg:
            for corruption in col_cfg["corruptions"]:
                if corruption["type"] not in corruptors_dict:
                    raise ValueError(
                        f"Unknown corruption function type: {corruption['type']}"
                    )
                corruption_fn = corruptors_dict[corruption["type"]]
                value = corruption_fn(
                    value, row_index=row_index, **corruption.get("args", {})
                )

        row[column_name] = value
    return row


def generate_corruptions(info, row, row_index, corruptors_dict):
    for corruption in info.get("corruptions", []):
        row = row.copy()  # Avoid modifying the original row for subsequent corruptions
        if corruption["type"] not in corruptors_dict:
            raise ValueError(f"Unknown corruption function type: {corruption['type']}")
        func = corruptors_dict[corruption["type"]]
        row = func(row, row_index=row_index, **corruption.get("args", {}))
    return row


def generate_linked_columns(table_cfg, row, output_dir):
    for _, col_info in table_cfg["linked_columns"].items():
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
            selected_idx = random.randint(0, len(linked_cols) - 1)
            selected_row = linked_cols.iloc[[selected_idx]].copy()
        else:
            raise ValueError(f"Unknown linked type: {linked_type}")
        # Insert column to row
        if rename_to:
            selected_row = selected_row.rename(columns=dict(zip(linked_on, rename_to)))

        row.update(selected_row.iloc[0])

    return row


def generate_tables(cfg, output_dir, generators_dict, corruptors_dict):
    # Iterate through each file and its corresponding configuration
    for table_name, table_cfg in cfg["data"].items():
        rows = []
        for i in range(table_cfg["N"]):
            row = {}
            row = generate_rows(table_cfg, row, i, generators_dict, corruptors_dict)
            row = generate_corruptions(table_cfg, row, i, corruptors_dict)
            rows.append(row)

        df = pd.DataFrame(rows).convert_dtypes()
        df.to_csv(output_dir / f"{table_name}.csv", index=False)

    for table_name, table_cfg in cfg.get("linked_data", {}).items():
        rows = []
        for i in range(table_cfg["N"]):
            row = {}
            row = generate_rows(table_cfg, row, i, generators_dict, corruptors_dict)
            row = generate_linked_columns(table_cfg, row, output_dir)
            row = generate_corruptions(table_cfg, row, i, corruptors_dict)
            rows.append(row)

        df = pd.DataFrame(rows).convert_dtypes()
        df.to_csv(output_dir / f"{table_name}.csv", index=False)


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

    with open(root / "configs" / args.config) as f:
        cfg = yaml.safe_load(f)
    output_dir = root / cfg["paths"]["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    generators_dict = {
        name: obj
        for name, obj in inspect.getmembers(generators)
        if inspect.isfunction(obj)
    }
    corruptors_dict = {
        name: obj
        for name, obj in inspect.getmembers(corruptors)
        if inspect.isfunction(obj)
    }

    generate_tables(cfg, output_dir, generators_dict, corruptors_dict)
