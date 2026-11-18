import logging
import logging.handlers
import multiprocessing
import polars as pl
from ehr2meds.preMEDS.constants import MAPPED_ID, SUBJECT_ID
from ehr2meds.preMEDS.data_handler import DataHandler
from ehr2meds.preMEDS.data_quality import (
    format_data_quality_summary,
    get_data_quality_summary,
)
from ehr2meds.preMEDS.processors import Processor
from tqdm import tqdm
from typing import Optional

logger = logging.getLogger(__name__)


def process_single_table_worker(args):
    """
    Standalone worker function to process a single table in a separate process.
    """
    table_name, table_config, output_path, write_file_type, chunksize, test, subject_id_mapping = args
    logger.info(f"Starting process for table: {table_name}")

    try:
        # Initialize handlers inside the worker process to avoid shared resource/file handle issues
        data_handler = DataHandler(
            output_dir=output_path,
            write_file_type=write_file_type,
            chunksize=chunksize,
        )
        processor = Processor()
        summary = []
        next_row_idx = 0

        for chunk in tqdm(
            data_handler.load_chunks(table_config["filename"], cols=table_config["columns"]),
            desc=f"Chunks {table_name}",
            position=0,
            leave=True,
        ):
            processed_chunk, row_result = processor.process(
                chunk,
                table_config,
                data_handler,
                subject_id_mapping,
                row_index_start=next_row_idx,
            )

            next_row_idx += len(chunk)
            data_handler.save(processed_chunk, table_name)

            # Only counts go back to the parent; sending the chunk would hold the whole table in its memory.
            null_counts = processed_chunk.null_count().row(0, named=True)
            summary.append([table_config["filename"], row_result, null_counts, processed_chunk.height])

            if test:
                break

        logger.info(f"Finished processing table: {table_name}. Save path {output_path}/{table_name}")

        return summary

    except Exception as e:
        logger.error(f"Error processing {table_name}: {str(e)}")
        raise

    finally:
        data_handler.close()


def init_worker_logging(queue, level: int) -> None:
    """spawn workers start without the parent's logging setup, so send their records to the parent."""
    root = logging.getLogger()
    root.handlers = [logging.handlers.QueueHandler(queue)]
    root.setLevel(level)


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

    def get_subject_id_mapping(self) -> Optional[pl.DataFrame]:
        if not self.cfg.get("subject_id_mapping"):
            return None
        # Load existing mapping if available
        logger.info("Loading dataframe for subject ID mapping")
        id_col = self.cfg.subject_id_mapping.subject_id_col
        map_col = self.cfg.subject_id_mapping.mapping_id_col
        cols = {id_col: None} | ({map_col: None} if map_col is not None else {})
        df = (
            self.data_handler.load(self.cfg.subject_id_mapping.file, cols=cols)
            .drop_nulls(id_col)
            .unique(subset=[id_col], keep="first")
            .with_columns(pl.col(id_col).cast(pl.String))
            # Always sort by the ID column for consistent mapping
            .sort(id_col)
        )
        logger.info(f"Number of patients in dataframe: {df.height}")

        # If no mapping column is provided, create a int-based mapping
        if not map_col:
            map_col = "mapping"
            df = df.with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias(map_col))

        # Save the mapping for reference.
        self.data_handler.save(df, "subject_id_mapping")

        return df.select(pl.col(id_col).alias(SUBJECT_ID), pl.col(map_col).cast(pl.Int64).alias(MAPPED_ID))

    def process_tables(self, subject_id_mapping: Optional[pl.DataFrame] = None) -> None:
        """Processes each table in the tables config using parallel workers"""

        # Prepare arguments for the worker pool
        worker_tasks = [
            (
                table_name,
                table_config,
                self.cfg.paths.output,
                self.cfg.write_file_type,
                self.cfg.chunksize,
                self.cfg.test,
                subject_id_mapping,
            )
            for table_name, table_config in self.cfg["tables"].items()
        ]

        logger.info(f"Starting multiprocessing pool with {self.cfg.num_workers} workers. Test enabled: {self.cfg.test}")

        # spawn, not fork: forking after polars has started its thread pool can deadlock the workers.
        context = multiprocessing.get_context("spawn")
        log_queue = context.Queue()
        root = logging.getLogger()
        listener = logging.handlers.QueueListener(log_queue, *root.handlers, respect_handler_level=True)
        listener.start()
        try:
            with context.Pool(
                processes=self.cfg.num_workers, initializer=init_worker_logging, initargs=(log_queue, root.level)
            ) as pool:
                worker_results = pool.map(process_single_table_worker, worker_tasks)
        finally:
            listener.stop()

        results = get_data_quality_summary(worker_results)

        if results is not None:
            print(format_data_quality_summary(results))
        else:
            logger.warning("Data-quality summary skipped due to tracking failure.")
