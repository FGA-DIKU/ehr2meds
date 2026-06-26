import pandas as pd
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.utils import (
    apply_mapping,
    apply_value_map,
    clean_data,
    map_pids_to_ints,
    validate_subject_id,
)
from pathlib import Path
from typing import Dict, Optional


class Processor:
    @staticmethod
    def process(
        df: pd.DataFrame,
        table_config: dict,
        data_handler: "DataHandler",
        subject_id_mapping: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Process the table.
        3. Apply value mappings
        4. apply columns map
        5. Pad values
        6. fill missing values
        7. combine datetime columns
        8. unroll columns (process codes)
        9. convert numeric columns
        10. apply pid integer mapping
        11. clean data
        12. validate subject_id column
        """
        df = apply_value_map(df, table_config)
        df = Processor._apply_mappings(df, table_config, data_handler)
        if subject_id_mapping is not None:
            df = map_pids_to_ints(df, subject_id_mapping)
        df = clean_data(df)
        validate_subject_id(df)

        return df

    @staticmethod
    def _get_mapping_table(data_handler, mapping):
        "Find path and relevant columns in either registry or the resources folder."
        filename = mapping["via_file"]
        cols = {mapping["join_on"]: None, mapping["target_column"]: None}

        register_path = Path(filename)
        if not register_path.exists():
            filename = str(Path(__file__).parent.parent / "resources" / filename)

        return data_handler.load(filename, cols=cols)

    @staticmethod
    def _apply_mappings(df: pd.DataFrame, table_config: dict, data_handler: DataHandler) -> pd.DataFrame:
        if table_config.get("mappings"):
            for mapping in table_config.mappings:
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