import ehr2meds.synthetic_data_generation.corrupt as corruptors
import ehr2meds.synthetic_data_generation.generate as generators
import hydra
import inspect
import pandas as pd
import random
from dotenv import load_dotenv
from ehr2meds.paths import get_config_path
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

load_dotenv()


def handle_mix_function(call_args, generators_dict):
    callable_args = call_args.copy()
    for func_cfg in callable_args["functions"]:
        if func_cfg["type"] in generators_dict:
            func_cfg["func"] = generators_dict[func_cfg["type"]]
        else:
            raise ValueError(f"Function {func_cfg['type']} not found in generators_dict")
    return callable_args


def generate_rows(table_cfg, row, row_index, generators_dict, corruptors_dict):
    for column_name, col_cfg in table_cfg["columns"].items():
        if col_cfg["type"] not in generators_dict:
            raise ValueError(f"Unknown generation function type: {col_cfg['type']}")
        func = generators_dict[col_cfg["type"]]

        col_args = col_cfg.get("args", {})
        if OmegaConf.is_config(col_args):
            col_args = OmegaConf.to_container(col_args, resolve=True)
        else:
            col_args = dict(col_args)
        if "by" in col_cfg:
            col_args = col_args[row[col_cfg["by"]]]
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
                    raise ValueError(f"Unknown corruption function type: {corruption['type']}")
                corruption_fn = corruptors_dict[corruption["type"]]
                value = corruption_fn(value, row_index=row_index, **corruption.get("args", {}))

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


def generate_linked_columns(table_cfg, row, output_dir, unused_idxs=None):
    for col_name, col_info in table_cfg["linked_columns"].items():
        linked_file = col_info["file"]
        linked_on = col_info["linked_on"]
        linked_type = col_info["type"]

        # Load the linked DataFrame from CSV file
        linked_file_path = output_dir / f"{linked_file}.csv"
        if not linked_file_path.exists():
            raise ValueError(
                f"Linked file '{linked_file_path}' not found. Make sure it's generated in the 'data' section first."
            )
        linked_df = pd.read_csv(linked_file_path)

        if linked_on not in linked_df.columns:
            raise ValueError(
                f"Linked columns {linked_on} not found in linked file '{linked_file}'. "
                f"Available columns: {list(linked_df.columns)}"
            )

        # Get columns
        linked_cols = linked_df[linked_on]
        if linked_type == "choice":
            selected_idx = random.randint(0, len(linked_cols) - 1)
            selected_row = linked_cols.iloc[[selected_idx]].copy()
        elif linked_type == "choice_unique":
            if unused_idxs is None:
                unused_idxs = linked_df.index.tolist()  # Initialize with all indices of the linked DataFrame
                random.shuffle(unused_idxs)  # Shuffle indices to ensure random selection
            selected_idx = unused_idxs.pop()
            selected_row = linked_cols.loc[[selected_idx]].copy()
        else:
            raise ValueError(f"Unknown linked type: {linked_type}")

        row.update({col_name: selected_row.item()})

    return row, unused_idxs


def save_df(df, output_dir, table_name, ext, saving_cfg):
    if saving_cfg.get("file_type"):
        ext = saving_cfg["file_type"]
    assert ext in ["csv", "asc", "parquet"]
    if ext == "parquet":
        df.to_parquet(output_dir / f"{table_name}.{ext}", index=False)
    else:
        sep = saving_cfg.get("sep", ",")
        encoding = saving_cfg.get("encoding", None)
        df.to_csv(
            output_dir / f"{table_name}.{ext}",
            index=False,
            sep=sep,
            encoding=encoding,
        )


def generate_correlated_cohort(cohort_cfg, generators_dict, corruptors_dict, output_dir):
    """Generate separate tables that share unique linked IDs and ordered conditioning."""
    linked_names = list(cohort_cfg["linked_columns"].keys())
    table_rows = {name: [] for name in cohort_cfg["tables"]}
    unused_idxs = None

    for i in range(cohort_cfg["N"]):
        context = {}
        context, unused_idxs = generate_linked_columns(
            cohort_cfg, context, output_dir, unused_idxs=unused_idxs
        )
        linked_values = {name: context[name] for name in linked_names}

        for table_name, table_spec in cohort_cfg["tables"].items():
            for child_idx in range(int(table_spec.get("repeat", 1))):
                row = generate_rows(
                    table_spec,
                    dict(context),
                    i * 1000 + child_idx,
                    generators_dict,
                    corruptors_dict,
                )
                generated = {column: row[column] for column in table_spec["columns"]}
                generated = generate_corruptions(table_spec, generated, i, corruptors_dict)
                table_rows[table_name].append({**linked_values, **generated})
                if child_idx == 0:
                    context.update(generated)

    return table_rows


def generate_tables(cfg, output_dir, generators_dict, corruptors_dict):
    # Iterate through each file and its corresponding configuration
    for table_name, table_cfg in cfg.get("data", {}).items():
        rows = []
        for i in range(table_cfg["N"]):
            row = {}
            row = generate_rows(table_cfg, row, i, generators_dict, corruptors_dict)
            row = generate_corruptions(table_cfg, row, i, corruptors_dict)
            rows.append(row)

        df = pd.DataFrame(rows).convert_dtypes()
        save_df(df, output_dir, table_name, ext=cfg.get("save_file_type", "csv"), saving_cfg=table_cfg.get("save_info", {}))

    for table_name, table_cfg in cfg.get("linked_data", {}).items():
        rows = []
        unused_idxs = None  # Reset unused indices for each linked table to ensure uniqueness within that table
        for i in range(table_cfg["N"]):
            row = {}
            row = generate_rows(table_cfg, row, i, generators_dict, corruptors_dict)
            row, unused_idxs = generate_linked_columns(table_cfg, row, output_dir, unused_idxs=unused_idxs)
            row = generate_corruptions(table_cfg, row, i, corruptors_dict)
            rows.append(row)

        df = pd.DataFrame(rows).convert_dtypes()
        save_df(df, output_dir, table_name, ext=cfg.get("save_file_type", "csv"), saving_cfg=table_cfg.get("save_info", {}))

    for _, cohort_cfg in cfg.get("correlated_data", {}).items():
        table_rows = generate_correlated_cohort(
            cohort_cfg, generators_dict, corruptors_dict, output_dir
        )
        for table_name, rows in table_rows.items():
            df = pd.DataFrame(rows).convert_dtypes()
            save_df(
                df,
                output_dir,
                table_name,
                ext=cfg.get("save_file_type", "csv"),
                saving_cfg=cohort_cfg.get("save_info", {}),
            )


@hydra.main(
    config_path=get_config_path(),
    config_name="root_config",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    random.seed(0)
    generators.np.random.seed(0)
    output_dir = Path(cfg.paths.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators_dict = {name: obj for name, obj in inspect.getmembers(generators) if inspect.isfunction(obj)}
    corruptors_dict = {name: obj for name, obj in inspect.getmembers(corruptors) if inspect.isfunction(obj)}

    generate_tables(cfg, output_dir, generators_dict, corruptors_dict)


if __name__ == "__main__":
    main()
