import logging
import pickle
from ehr2meds.preMEDS.concept_processors import (
    RegisterConceptProcessor,
    SPConceptProcessor,
)
from ehr2meds.preMEDS.constants import SUBJECT_ID
from ehr2meds.preMEDS.data_handler import DataHandler
from tqdm import tqdm
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class PREMEDSExtractor:
    """
    Preprocessor for MEDS (Medical Event Data Set) that handles patient data and medical concepts.

    This class processes medical data by:
    1. Building subject ID mappings
    2. Processing various medical concepts (diagnoses, procedures, etc.)
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

        # Create data handler for concepts
        self.data_handler = DataHandler(
            output_dir=cfg.paths.output,
            file_type=cfg.write_file_type,
            path=cfg.paths.concepts,
            chunksize=self.chunksize,
            test_rows=cfg.get("test_rows", 1_000_000),
            test=cfg.test,
        )
        if cfg.get("register_concepts"):
            # Create data handler for register concepts
            self.register_data_handler = DataHandler(
                output_dir=cfg.paths.output,
                file_type=cfg.write_file_type,
                path=cfg.paths.register_concepts,
                chunksize=self.chunksize,
                test_rows=cfg.get("test_rows", 1_000_000),
                test=cfg.test,
            )

        self.concept_processor = SPConceptProcessor()
        self.register_concept_processor = RegisterConceptProcessor()

    def __call__(self):
        subject_id_mapping = self.get_subject_id_mapping()
        if self.cfg.get("register_concepts"):
            self.format_register_concepts(subject_id_mapping)
        self.format_concepts(subject_id_mapping)

    def get_subject_id_mapping(self) -> Union[None, Dict[str, int]]:
        if not self.cfg.get("subject_id_mapping"):
            return None
        # Load existing mapping if available
        logger.info("Loading dataframe for subject ID mapping")
        id_col = self.cfg.subject_id_mapping.subject_id_col
        df = (
            self.data_handler.load_pandas(
                self.cfg.subject_id_mapping.filename,
                cols=[id_col],
            )
            .rename(columns={id_col: SUBJECT_ID})
            .dropna(subset=[SUBJECT_ID], how="any")
            .drop_duplicates(subset=[SUBJECT_ID])
        )
        logger.info(f"Number of patients in dataframe: {len(df)}")

        hash_to_int_map = dict(zip(df[SUBJECT_ID], range(len(df))))

        # Save the mapping for reference.
        with open(f"{self.cfg.paths.output}/hash_to_integer_map.pkl", "wb") as f:
            pickle.dump(hash_to_int_map, f)

        return hash_to_int_map

    def format_concepts(self, subject_id_mapping: Optional[Dict[str, int]]) -> None:
        """Process all medical concepts"""
        for concept_type, concept_config in self.cfg.get("concepts", {}).items():
            try:
                self._process_concept_chunks(
                    concept_type,
                    concept_config,
                    subject_id_mapping,
                    self.time_stamp_dict,
                )
            except Exception as e:
                logger.warning(f"Error processing {concept_type}: {str(e)}")

    def format_register_concepts(self, subject_id_mapping: Optional[Dict[str, int]]) -> None:
        """Process the register concepts using the register-specific data handler"""

        for concept_type, concept_config in self.cfg.get("register_concepts", {}).items():
            logger.info(f"Processing register concept: {concept_type}")
            try:
                self.process_register_concept_chunks(
                    concept_type,
                    concept_config,
                    subject_id_mapping,
                )
            except Exception as e:
                logger.warning(f"Error processing {concept_type}: {str(e)}")

    def process_register_concept_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Optional[Dict[str, int]],
    ) -> None:
        first_chunk = True
        for chunk in tqdm(
            self.register_data_handler.load_chunks(concept_config),
            desc=f"Chunks {concept_type}",
        ):
            processed_chunk = self.register_concept_processor.process(
                chunk,
                concept_config,
                subject_id_mapping,
                self.register_data_handler,
            )

            self._safe_save(self.register_data_handler, processed_chunk, concept_type, first_chunk)
            first_chunk = False

    def _process_concept_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Optional[Dict[str, int]],
        time_stamp_dict: Optional[dict] = None,
    ) -> None:
        first_chunk = True
        for chunk in tqdm(
            self.data_handler.load_chunks(concept_config),
            desc=f"Chunks {concept_type}",
        ):
            processed_chunk = self.concept_processor.process(chunk, concept_config, subject_id_mapping, time_stamp_dict)
            self._safe_save(self.data_handler, processed_chunk, concept_type, first_chunk)
            first_chunk = False

    def _safe_save(self, data_handler, processed_chunk, concept_type, first_chunk: bool) -> None:
        if not processed_chunk.empty:
            mode = "w" if first_chunk else "a"
            data_handler.save(processed_chunk, concept_type, mode=mode)
        else:
            logger.warning(f"Empty processed chunk for {concept_type}, skipping save")
