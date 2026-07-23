import logging
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.processors import Processor
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


def process_single_table_worker(args):
    """
    Standalone worker function to process a single table in a separate process.
    """
    table_name, table_config, output_path, write_file_type, chunksize, subject_id_mapping = args
    logger.info(f"Starting process for table: {table_name}")

    try:
        # Initialize handlers inside the worker process to avoid shared resource/file handle issues
        data_handler = DataHandler(
            output_dir=output_path,
            write_file_type=write_file_type,
            chunksize=chunksize,
        )
        processor = Processor()

        for chunk in tqdm(
            data_handler.load_chunks(table_config["filename"], cols=table_config["columns"]),
            desc=f"Chunks {table_name}",
            position=0,  # Helps prevent progress bars from overlapping wildly
            leave=True,
        ):
            processed_chunk = processor.process(
                chunk,
                table_config,
                data_handler,
                subject_id_mapping,
            )

            data_handler.save(processed_chunk, table_name)

        logger.info(f"Finished processing table: {table_name}. Save path {output_path}/{table_name}")
    except Exception as e:
        logger.error(f"Error processing {table_name}: {str(e)}")
        raise


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
        # Create data handler for tables
        self.data_handler = DataHandler(
            output_dir=cfg.paths.output,
            write_file_type=cfg.write_file_type,
            chunksize=cfg.chunksize,
        )
        self.processor = Processor()

    def __call__(self):
        subject_id_mapping = self.get_subject_id_mapping()
        self.process_tables(subject_id_mapping)

    def get_subject_id_mapping(self) -> Union[None, Dict[str, int]]:
        if not self.cfg.get("subject_id_mapping"):
            return None
        # Load existing mapping if available
        logger.info("Loading dataframe for subject ID mapping")
        id_col = self.cfg.subject_id_mapping.subject_id_col
        map_col = self.cfg.subject_id_mapping.mapping_id_col
        cols = {id_col: None} | ({map_col: None} if map_col is not None else {})
        df = (
            self.data_handler.load(self.cfg.subject_id_mapping.file, cols=cols)
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

    def process_tables(self, subject_id_mapping: Optional[Dict[str, int]] = None) -> None:
        """Processes each table in the tables config using parallel workers"""

        # Prepare arguments for the worker pool
        worker_tasks = [
            (table_name, table_config, self.cfg.paths.output, self.cfg.write_file_type, self.cfg.chunksize, subject_id_mapping)
            for table_name, table_config in self.cfg["tables"].items()
        ]

        logger.info(f"Starting multiprocessing pool with {self.cfg.num_workers} workers.")

        # Use a Pool to manage the N concurrent table workers
        with Pool(processes=self.cfg.num_workers) as pool:
            # map will block until all tables are processed and raise exceptions if any fail
            pool.map(process_single_table_worker, worker_tasks)

    def process_table_chunks(
        self,
        table_name: str,
        table_config: dict,
        subject_id_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        # Chunks are yielded in source-file order. Advancing by the number of
        # input rows makes row_idx independent of rows removed during processing.
        next_row_idx = 0
        for chunk in tqdm(
            self.data_handler.load_chunks(table_config["filename"], cols=table_config["columns"]),
            desc=f"Chunks {table_name}",
        ):
            input_row_count = len(chunk)
            processed_chunk = self.processor.process(
                chunk,
                table_config,
                self.data_handler,
                subject_id_mapping,
                row_index_start=next_row_idx,
            )

            next_row_idx += input_row_count

            self.data_handler.save(processed_chunk, table_name)

            if self.cfg.get("test"):
                break
