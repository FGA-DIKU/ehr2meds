import pandas as pd
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.mappings import MAPPING_STRATEGIES, apply_mapping
from ehr2meds.preMEDS.timestamps import add_timestamp_columns
from ehr2meds.preMEDS.utils import (
    add_row_idx,
    apply_value_map,
    clean_data,
    map_pids_to_ints,
    normalize_code_columns,
    normalize_integer_columns,
    remove_timezones,
    validate_subject_id,
)
from pathlib import Path
from typing import Dict, List, Optional
from ehr2meds.preMEDS.row_tracking import RowTracker


class Processor:
    @staticmethod
    def process(
        df,
        table_config,
        data_handler,
        subject_id_mapping=None,
        row_index_start=0,
        track_rows=False,
    ):
        """Process the table.

        1. Add row index to input tables
        2. Remove timezone information from timezone-aware datetime columns
        3. OPTIONAL: Apply table mappings
        4. OPTIONAL: Normalize integer columns
        5. OPTIONAL: Apply pid integer mapping
        6. Normalize string columns
        7. OPTIONAL: Apply value mappings
        8. OPTIONAL: Construct timestamps from date and time columns
        9. Clean data
        10. Validate subject_id column
        """
        tracker = RowTracker(initial_rows=len(df)) if track_rows else None

        df = add_row_idx(df, start=row_index_start)
        df = remove_timezones(df)

        df = Processor.apply_mappings(
            df,
            table_config.get("mappings", []),
            data_handler,
        )
        if tracker is not None:
            tracker.checkpoint("Table mappings", df)

        df = normalize_integer_columns(
            df,
            table_config.get("normalize_integer_columns", []),
        )

        if subject_id_mapping is not None:
            df = map_pids_to_ints(df, subject_id_mapping)
            if tracker is not None:
                tracker.checkpoint("Subject ID mapping", df)

        df = normalize_code_columns(df)

        df = apply_value_map(
            df,
            table_config.get("value_map", {}),
        )

        df = add_timestamp_columns(
            df,
            table_config.get("timestamp_columns", {}),
        )

        df = clean_data(df, tracker=tracker)

        validate_subject_id(df)

        if tracker is not None:
            return df, tracker.result()

        return df

    @staticmethod
    def apply_mappings(df: pd.DataFrame, mapping_cfg: List[dict], data_handler: DataHandler) -> pd.DataFrame:
        for mapping in mapping_cfg:
            strategy_name = mapping.get("function")
            strategy = MAPPING_STRATEGIES.get(strategy_name)
            if strategy_name is not None and strategy is None:
                available = ", ".join(sorted(MAPPING_STRATEGIES))
                raise ValueError(f"Unknown mapping function {strategy_name!r}. Available functions: {available}")

            if strategy is not None:
                mapping_function, mapping_columns = strategy
                map_table = Processor.get_mapping_table(data_handler, mapping, mapping_columns)
                df = mapping_function(df, map_table, mapping)
                continue

            target_columns = mapping["target_columns"]
            map_table = Processor.get_mapping_table(data_handler, mapping, tuple(target_columns))
            df = apply_mapping(
                df,
                map_table,
                join_col=mapping["join_on"],
                source_col=mapping["source_column"],
                target_columns=target_columns,
                how=mapping.get("how", "inner"),
                drop_source=mapping.get("drop_source", False),
            )
        return df

    @staticmethod
    def get_mapping_table(
        data_handler: DataHandler,
        mapping: dict,
        target_columns: tuple[str, ...],
    ):
        """Load the columns required by a standard or specialized mapping."""
        filename = Path(mapping["via_file"])
        if not filename.exists():
            filename = Path(__file__).parents[2] / "resources" / filename

        cols = dict.fromkeys((mapping["join_on"], *target_columns))

        return data_handler.load(str(filename), cols=cols)
