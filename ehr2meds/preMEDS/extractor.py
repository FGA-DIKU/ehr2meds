import logging
import pickle
from ehr2meds.preMEDS.constants import SUBJECT_ID
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.processors import Processor
from ehr2meds.preMEDS.utils import (
    factorize_subject_id,
    select_and_rename_columns,
)
from tqdm import tqdm
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PREMEDSExtractor:
    """
    Preprocessor for MEDS (Medical Event Data Set) that handles patient data and medical tables.

    This class processes medical data by:
    1. Building subject ID mappings
    2. Processing various medical tables (diagnoses, procedures, etc.)
    3. Formatting and cleaning the data according to specified configurations
    """

    def __init__(self, cfg):
        self.cfg = cfg
        logger.info(f"test {cfg.test}")
        self.chunksize = cfg.get("chunksize", 500_000)
        if cfg.get("align_timestamps"):
            self.time_stamp_dict = {
                "names": cfg.align_timestamps.names,
                "format": cfg.align_timestamps.format,
            }
        else:
            self.time_stamp_dict = None

        # Create data handler for tables
        self.data_handler = DataHandler(
            output_dir=cfg.paths.output,
            file_type=cfg.write_file_type,
            chunksize=self.chunksize,
            test_rows=cfg.get("test_rows", 1_000_000),
            test=cfg.test,
        )
        self.processor = Processor()

    def __call__(self):
        subject_id_mapping = self.format_patients_info()
        self.format_tables(subject_id_mapping)

    def format_patients_info(self) -> Dict[str, int]:
        """
        Load and process patient information, creating a mapping of patient IDs.

        Returns:
            Dict[str, int]: Mapping from original patient IDs to integer IDs
        """
        logger.info("Load patients info")
        df = self.data_handler.load_pandas(
            self.cfg.patients_info.filename,
            cols=list(self.cfg.patients_info.get("rename_columns", {}).keys()),
            **self.cfg.patients_info.get("file_info", {}),
        )
        # Use columns_map to subset and rename the columns.
        df = select_and_rename_columns(df, self.cfg.patients_info.get("rename_columns", {}))
        logger.info(f"Number of patients after selecting columns: {len(df)}")

        df, hash_to_int_map = factorize_subject_id(df)
        # Save the mapping for reference.
        with open(f"{self.cfg.paths.output}/hash_to_integer_map.pkl", "wb") as f:
            pickle.dump(hash_to_int_map, f)

        df = df.dropna(subset=[SUBJECT_ID], how="any")
        logger.info(f"Number of patients before saving: {len(df)}")
        self.data_handler.save(df, "subject")

        return hash_to_int_map

    def format_tables(self, subject_id_mapping: Dict[str, int]) -> None:
        """Process the tables using the data handler"""
        for table_type, table_config in self.cfg.get("tables", {}).items():
            logger.info(f"Processing table: {table_type}")
            try:
                self.process_table_chunks(
                    table_type,
                    table_config,
                    subject_id_mapping,
                    self.time_stamp_dict,
                )
            except Exception as e:
                logger.warning(f"Error processing {table_type}: {str(e)}")

    def process_table_chunks(
        self,
        table_type: str,
        table_config: dict,
        subject_id_mapping: Dict[str, int],
        time_stamp_dict: Optional[dict] = None,
    ) -> None:
        first_chunk = True
        for chunk in tqdm(
            self.data_handler.load_chunks(table_config),
            desc=f"Chunks {table_type}",
        ):
            processed_chunk = self.processor.process(
                chunk,
                table_config,
                subject_id_mapping,
                self.data_handler,
                time_stamp_dict,
            )

            self._safe_save(self.data_handler, processed_chunk, table_type, first_chunk)
            first_chunk = False

    def _safe_save(self, data_handler, processed_chunk, table_type, first_chunk: bool) -> None:
        if not processed_chunk.empty:
            mode = "w" if first_chunk else "a"
            data_handler.save(processed_chunk, table_type, mode=mode)
        else:
            logger.warning(f"Empty processed chunk for {table_type}, skipping save")
