import ehr2meds.synthetic_data_generation.corrupt as corruptors
import ehr2meds.synthetic_data_generation.generate as generators
import functools
import hydra
import inspect
import polars as pl
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

        col_args = dict(col_cfg.get("args", {}))  # plain dict, so "match" can add keys
        if col_cfg["type"] == "mix_function":
            col_args = handle_mix_function(OmegaConf.to_container(col_cfg["args"]), generators_dict)

        # Handle dependencies between columns using the "match" key
        for keyword, match_col in col_cfg.get("match", {}).items():
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


@functools.cache
def load_linked_table(path):
    return pl.read_csv(path, infer_schema_length=None)


def generate_linked_columns(table_cfg, row, row_index, output_dir, corruptors_dict, unused_idxs=None):
    # All columns linked to the same file are read from the same row of it, picked by the first such column's type
    selected_idxs = {}
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
        linked_df = load_linked_table(linked_file_path)

        if linked_on not in linked_df.columns:
            raise ValueError(
                f"Linked columns {linked_on} not found in linked file '{linked_file}'. "
                f"Available columns: {list(linked_df.columns)}"
            )

        if linked_file not in selected_idxs:
            if linked_type == "choice":
                selected_idxs[linked_file] = random.randint(0, linked_df.height - 1)
            elif linked_type == "choice_unique":
                if unused_idxs is None:
                    unused_idxs = list(range(linked_df.height))  # Initialize with all row indices of the linked DataFrame
                    random.shuffle(unused_idxs)  # Shuffle indices to ensure random selection
                selected_idxs[linked_file] = unused_idxs.pop()
            else:
                raise ValueError(f"Unknown linked type: {linked_type}")

        value = linked_df.item(selected_idxs[linked_file], linked_on)

        # Apply column-specific corruptions if specified
        if "corruptions" in col_info:
            for corruption in col_info["corruptions"]:
                if corruption["type"] not in corruptors_dict:
                    raise ValueError(f"Unknown corruption function type: {corruption['type']}")
                corruption_fn = corruptors_dict[corruption["type"]]
                value = corruption_fn(value, row_index=row_index, **corruption.get("args", {}))

        row.update({col_name: value})

    return row, unused_idxs


def hidden_columns(table_cfg):
    """Columns marked hidden: true, which other columns can match on but are not saved."""
    hidden = []
    for section in ["columns", "linked_columns"]:
        for col_name, col_cfg in table_cfg.get(section, {}).items():
            if col_cfg.get("hidden", False):
                hidden.append(col_name)
    return hidden


def rows_to_df(rows):
    """DataFrame of the generated rows. Columns of mixed types, e.g. times with "." corruptions, become text."""
    columns = {}
    for col_name in rows[0]:
        values = [row[col_name] for row in rows]
        types = {type(value) for value in values if value is not None}
        if len(types) > 1:
            values = [None if value is None else str(value) for value in values]
        columns[col_name] = values
    return pl.DataFrame(columns)


def save_df(df, output_dir, table_name, ext, saving_cfg):
    if saving_cfg.get("file_type"):
        ext = saving_cfg["file_type"]
    assert ext in ["csv", "asc", "parquet"]
    path = output_dir / f"{table_name}.{ext}"
    if ext == "parquet":
        df.write_parquet(path)
    else:
        # Dates and times as str() writes them (e.g. "12:30:00", no zero microseconds), empty strings as empty fields
        df = df.with_columns(pl.col(pl.Date, pl.Datetime, pl.Time).map_elements(str, return_dtype=pl.String))
        df = df.with_columns(pl.col(pl.String).replace("", None))
        sep = saving_cfg.get("sep", ",")
        encoding = saving_cfg.get("encoding", "utf-8")
        path.write_text(df.write_csv(separator=sep), encoding=encoding)


def generate_tables(cfg, output_dir, generators_dict, corruptors_dict):
    # Iterate through each file and its corresponding configuration
    for table_name, table_cfg in cfg.get("data", {}).items():
        rows = []
        for i in range(table_cfg["N"]):
            row = {}
            row = generate_rows(table_cfg, row, i, generators_dict, corruptors_dict)
            row = generate_corruptions(table_cfg, row, i, corruptors_dict)
            rows.append(row)

        df = rows_to_df(rows)
        df = df.drop(hidden_columns(table_cfg))
        save_df(df, output_dir, table_name, ext=cfg.get("save_file_type", "csv"), saving_cfg=table_cfg.get("save_info", {}))
        load_linked_table.cache_clear()

    for table_name, table_cfg in cfg.get("linked_data", {}).items():
        rows = []
        unused_idxs = None  # Reset unused indices for each linked table to ensure uniqueness within that table
        for i in range(table_cfg["N"]):
            row = {}
            # Linked columns first, so columns can match on them
            row, unused_idxs = generate_linked_columns(table_cfg, row, i, output_dir, corruptors_dict, unused_idxs=unused_idxs)
            row = generate_rows(table_cfg, row, i, generators_dict, corruptors_dict)
            row = generate_corruptions(table_cfg, row, i, corruptors_dict)
            rows.append(row)

        df = rows_to_df(rows)
        df = df.drop(hidden_columns(table_cfg))
        save_df(df, output_dir, table_name, ext=cfg.get("save_file_type", "csv"), saving_cfg=table_cfg.get("save_info", {}))
        load_linked_table.cache_clear()


@hydra.main(
    config_path=get_config_path(),
    config_name="root_config",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    random.seed(0)
    output_dir = Path(cfg.paths.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generators_dict = {name: obj for name, obj in inspect.getmembers(generators) if inspect.isfunction(obj)}
    corruptors_dict = {name: obj for name, obj in inspect.getmembers(corruptors) if inspect.isfunction(obj)}

    generate_tables(cfg, output_dir, generators_dict, corruptors_dict)


if __name__ == "__main__":
    main()
