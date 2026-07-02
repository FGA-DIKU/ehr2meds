import pandas as pd
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.utils import (
    apply_mapping,
    apply_value_map,
    compose_columns,
    clean_data,
    map_pids_to_ints,
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
    ) -> pd.DataFrame:
        """Process the table.
        1. OPTIONAL: Apply value mappings
        2. OPTIONAL: Apply columns map
        3. OPTIONAL: Compose new columns from existing ones
        4. OPTIONAL: Apply pid integer mapping
        5. clean data
        6. validate subject_id column
        """
        df = Processor._apply_mappings(df, table_config.get("mappings", []), data_handler)
        df = apply_value_map(df, table_config.get("value_map", {}))
        df = compose_columns(df, table_config.get("compose", {}))
        if subject_id_mapping is not None:
            df = map_pids_to_ints(df, subject_id_mapping)
        df = clean_data(df)
        validate_subject_id(df)

        return df

    @staticmethod
    def _apply_mappings(df: pd.DataFrame, mapping_cfg: List[dict], data_handler: DataHandler) -> pd.DataFrame:
        for mapping in mapping_cfg:
            map_table = Processor._get_mapping_table(data_handler, mapping)
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
    def _get_mapping_table(data_handler, mapping: dict):
        "Find path and relevant columns in either registry or the resources folder."
        filename = mapping["via_file"]
        cols = dict.fromkeys([mapping["join_on"], mapping["target_column"]])

        register_path = Path(filename)
        if not register_path.exists():
            filename = str(Path(__file__).parent.parent / "resources" / filename)  # TODO: Seems very hacky

        return data_handler.load(filename, cols=cols)
