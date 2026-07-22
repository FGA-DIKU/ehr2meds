import pandas as pd
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.utils import (
    add_row_idx,
    apply_mapping,
    apply_value_map,
    clean_data,
    map_pids_to_ints,
    remove_timezones,
    validate_subject_id,
)
from pathlib import Path
from typing import Dict, List, Optional


class Processor:
    @staticmethod
    def process(
        df: pd.DataFrame,
        table_config: dict,
        data_handler: DataHandler,
        subject_id_mapping: Optional[Dict[str, int]] = None,
        row_index_start: int = 0,
    ) -> pd.DataFrame:
        """Process the table.
        1. Add row index to input tables
        2. OPTIONAL: Apply value mappings
        3. OPTIONAL: Apply columns map
        4. OPTIONAL: Apply pid integer mapping
        5. Remove timezone information from timezone-aware datetime columns
        6. Clean data
        7. Validate subject_id column
        """
        df = add_row_idx(df, start=row_index_start)
        df = apply_value_map(df, table_config.get("value_map", {}))
        df = Processor.apply_mappings(df, table_config.get("mappings", []), data_handler)
        if subject_id_mapping is not None:
            df = map_pids_to_ints(df, subject_id_mapping)
        df = remove_timezones(df)
        df = clean_data(df)
        validate_subject_id(df)
        return df

    @staticmethod
    def apply_mappings(df: pd.DataFrame, mapping_cfg: List[dict], data_handler: DataHandler) -> pd.DataFrame:
        for mapping in mapping_cfg:
            map_table = Processor.get_mapping_table(data_handler, mapping)
            df = apply_mapping(
                df,
                map_table,
                join_col=mapping["join_on"],
                source_col=mapping["source_column"],
                target_col=mapping["target_column"],
                rename_to=mapping["rename_to"],
                how=mapping.get("how", "inner"),
                drop_source=mapping.get("drop_source", False),
            )
        return df

    @staticmethod
    def get_mapping_table(data_handler, mapping: dict):
        "Find path and relevant columns in either registry or the resources folder."
        filename = mapping["via_file"]
        cols = dict.fromkeys([mapping["join_on"], mapping["target_column"]])

        register_path = Path(filename)
        if not register_path.exists():
            filename = str(Path(__file__).parent.parent / "resources" / filename)  # TODO: Seems very hacky

        return data_handler.load(filename, cols=cols)
