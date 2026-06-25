import logging
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.processors import Processor
from tqdm import tqdm
from typing import Dict, Optional, Union

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
        subject_id_mapping = self.get_subject_id_mapping()
        self.format_tables(subject_id_mapping)

    def get_subject_id_mapping(self) -> Union[None, Dict[str, int]]:
        if not self.cfg.get("subject_id_mapping"):
            return None
        # Load existing mapping if available
        logger.info("Loading dataframe for subject ID mapping")
        id_col = self.cfg.subject_id_mapping.subject_id_col
        map_col = self.cfg.subject_id_mapping.mapping_id_col
        df = (
            self.data_handler.load(
                self.cfg.subject_id_mapping.file,
                cols=[id_col] + ([map_col] if map_col else []),
            )
            .dropna(subset=[id_col], how="any")
            .drop_duplicates(subset=[id_col])
        )
        logger.info(f"Number of patients in dataframe: {len(df)}")
        if df[id_col].dtype != object:
            df[id_col] = df[id_col].astype(str)

        # Always sort by the ID column for consistent mapping
        df = df.sort_values(by=id_col).reset_index(drop=True)

        # If no mapping column is provided, create a int-based mapping
        if not map_col:
            df["mapping"] = df.index
            map_col = "mapping"
        subject_id_mapping = dict(zip(df[id_col], df[map_col]))

        # Save the mapping for reference.
        df.to_csv(f"{self.cfg.paths.output}/subject_id_mapping.csv", index=False)

        return subject_id_mapping

    def format_tables(self, subject_id_mapping: Optional[Dict[str, int]] = None) -> None:
        """Process the tables using the data handler"""
        for table_type, table_config in self.cfg.get("tables", {}).items():
            logger.info(f"Processing table: {table_type}")
            try:
                self.process_table_chunks(
                    table_type,
                    table_config,
                    subject_id_mapping,
                )
            except Exception as e:
                logger.warning(f"Error processing {table_type}: {str(e)}")

    def process_table_chunks(
        self,
        table_type: str,
        table_config: dict,
        subject_id_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        first_chunk = True
        for chunk in tqdm(
            self.data_handler.load_chunks(table_config),
            desc=f"Chunks {table_type}",
        ):
            processed_chunk = self.processor.process(
                chunk,
                table_config,
                self.data_handler,
                subject_id_mapping,
            )

            self._safe_save(self.data_handler, processed_chunk, table_type, first_chunk)
            first_chunk = False

    def _safe_save(self, data_handler, processed_chunk, table_type, first_chunk: bool) -> None:
        if not processed_chunk.empty:
            mode = "w" if first_chunk else "a"
            data_handler.save(processed_chunk, table_type, mode=mode)
        else:
            logger.warning(f"Empty processed chunk for {table_type}, skipping save")
