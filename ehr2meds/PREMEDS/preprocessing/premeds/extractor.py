import pickle
from typing import Dict, Optional, Iterator

import pandas as pd
from tqdm import tqdm

from ehr2meds.PREMEDS.preprocessing.constants import SUBJECT_ID
from ehr2meds.PREMEDS.preprocessing.io.data_handling import DataConfig, DataHandler
from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import (
    factorize_subject_id,
    select_and_rename_columns,
)
from ehr2meds.PREMEDS.preprocessing.premeds.registers import RegisterConceptProcessor
from ehr2meds.PREMEDS.preprocessing.premeds.sp import ConceptProcessor


class PREMEDSExtractor:
    """
    Preprocessor for MEDS (Medical Event Data Set) that handles patient data and medical concepts.

    This class processes medical data by:
    1. Building subject ID mappings
    2. Processing various medical concepts (diagnoses, procedures, etc.)
    3. Formatting and cleaning the data according to specified configurations
    """

    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.test = cfg.test
        self.logger.info(f"test {self.test}")
        self.initial_patients = set()
        self.formatted_patients = set()
        self.chunksize = cfg.get("chunksize", 500_000)
        # Create data handler for concepts
        self.data_handler = DataHandler(
            DataConfig(
                output_dir=cfg.paths.output_dir,
                file_type=cfg.paths.file_type,
                datastore=cfg.data_path.concepts.get("datastore"),
                dump_path=cfg.data_path.concepts.dump_path,
                chunksize=self.chunksize,
            ),
            logger,
            env=cfg.env,
            test=self.test,
        )
        if cfg.get("register_concepts"):
            # Create data handler for register concepts
            self.register_data_handler = DataHandler(
                DataConfig(
                    output_dir=cfg.paths.output_dir,
                    file_type=cfg.paths.file_type,
                    datastore=cfg.data_path.register_concepts.get("datastore"),
                    dump_path=cfg.data_path.register_concepts.dump_path,
                    chunksize=self.chunksize,
                ),
                logger,
                env=cfg.env,
                test=self.test,
            )

            # Create data handler for mappings
            self.link_file_handler = DataHandler(
                DataConfig(
                    output_dir=cfg.paths.output_dir,
                    file_type=cfg.paths.file_type,
                    datastore=cfg.data_path.pid_link.get("datastore"),
                    dump_path=cfg.data_path.pid_link.dump_path,
                ),
                logger,
                env=cfg.env,
                test=self.test,
            )

        self.concept_processor = ConceptProcessor()
        self.register_concept_processor = RegisterConceptProcessor()

    def __call__(self):
        subject_id_mapping = self.format_patients_info()
        if self.cfg.get("register_concepts"):
            self.format_register_concepts(subject_id_mapping)
        self.format_concepts(subject_id_mapping)

    def format_patients_info(self) -> Dict[str, int]:
        """
        Load and process patient information, creating a mapping of patient IDs.

        Returns:
            Dict[str, int]: Mapping from original patient IDs to integer IDs
        """
        self.logger.info("Load patients info")
        df = self.data_handler.load_pandas(
            self.cfg.patients_info.filename,
            cols=list(self.cfg.patients_info.get("rename_columns", {}).keys()),
        )
        # Use columns_map to subset and rename the columns.
        df = select_and_rename_columns(
            df, self.cfg.patients_info.get("rename_columns", {})
        )
        self.logger.info(f"Number of patients after selecting columns: {len(df)}")

        df, hash_to_int_map = factorize_subject_id(df)
        # Save the mapping for reference.
        with open(f"{self.cfg.paths.output_dir}/hash_to_integer_map.pkl", "wb") as f:
            pickle.dump(hash_to_int_map, f)

        df = df.dropna(subset=[SUBJECT_ID], how="any")
        self.logger.info(f"Number of patients before saving: {len(df)}")
        self.data_handler.save(df, "subject")

        return hash_to_int_map

    def format_concepts(self, subject_id_mapping: Dict[str, int]) -> None:
        """Process all medical concepts"""
        for concept_type, concept_config in self.cfg.concepts.items():
            try:
                if concept_type == "admissions":
                    self.format_admissions(concept_config, subject_id_mapping)
                    continue
                self._process_concept_chunks(
                    concept_type, concept_config, subject_id_mapping, first_chunk=True
                )
            except Exception as e:
                self.logger.warning(f"Error processing {concept_type}: {str(e)}")

    def format_register_concepts(self, subject_id_mapping: Dict[str, int]) -> None:
        """Process the register concepts using the register-specific data handler"""
        # Load the register-SP mapping once - this maps register PIDs to SP hashes
        register_sp_link = self._get_register_sp_link()

        for concept_type, concept_config in self.cfg.register_concepts.items():
            self.logger.info(f"Processing register concept: {concept_type}")
            try:
                self.process_register_concept_chunks(
                    concept_type,
                    concept_config,
                    subject_id_mapping,
                    register_sp_link,
                    first_chunk=True,
                )
            except Exception as e:
                self.logger.warning(f"Error processing {concept_type}: {str(e)}")

    def process_register_concept_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        register_sp_link: pd.DataFrame,
        first_chunk: bool,
    ) -> None:
        for chunk in tqdm(
            self.register_data_handler.load_chunks(concept_config),
            desc=f"Chunks {concept_type}",
        ):
            processed_chunk = self.register_concept_processor.process(
                chunk,
                concept_config,
                subject_id_mapping,
                self.register_data_handler,
                register_sp_link,
                join_link_col=self.cfg.data_path.pid_link.join_col,  #  for linking to sp data
                target_link_col=self.cfg.data_path.pid_link.target_col,  #  for linking to sp data
            )

            self._safe_save(
                self.register_data_handler, processed_chunk, concept_type, first_chunk
            )

    def _process_concept_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        first_chunk: bool,
    ) -> None:
        for chunk in tqdm(
            self.data_handler.load_chunks(concept_config),
            desc=f"Chunks {concept_type}",
        ):
            processed_chunk = self.concept_processor.process(
                chunk, concept_config, subject_id_mapping
            )
            self._safe_save(
                self.data_handler, processed_chunk, concept_type, first_chunk
            )

    def _safe_save(
        self, data_handler, processed_chunk, concept_type, first_chunk: bool
    ) -> None:
        if not processed_chunk.empty:
            mode = "w" if first_chunk else "a"
            data_handler.save(processed_chunk, concept_type, mode=mode)
            first_chunk = False
        else:
            self.logger.warning(
                f"Empty processed chunk for {concept_type}, skipping save"
            )

    def _get_register_sp_link(self) -> pd.DataFrame:
        pid_link_cfg = self.cfg.data_path.pid_link
        register_sp_link = self.link_file_handler.load_pandas(
            pid_link_cfg.filename, cols=[pid_link_cfg.join_col, pid_link_cfg.target_col]
        )
        return register_sp_link

    def format_admissions(
        self, admissions_config: dict, subject_id_mapping: Dict[str, int]
    ) -> None:
        """Process the admissions concept separately, handling patients across chunks."""
        first_chunk = True
        last_patient_data = None  # Store data for patient that spans chunks

        for chunk in tqdm(
            self.data_handler.load_chunks(admissions_config),
            desc="Chunks admissions",
        ):
            # Process the chunk with any carried over patient data
            processed_chunk, last_patient_data = (
                ConceptProcessor.process_adt_admissions(
                    chunk, admissions_config, subject_id_mapping, last_patient_data
                )
            )

            if not processed_chunk.empty:
                mode = "w" if first_chunk else "a"
                self.data_handler.save(processed_chunk, "admissions", mode=mode)
                first_chunk = False

        # Process any remaining last patient data
        if last_patient_data and last_patient_data["events"]:
            final_df = pd.DataFrame(last_patient_data["events"])
            if not final_df.empty:
                self.data_handler.save(final_df, "admissions", mode="a")

    def _load_csv_chunks(
        self, file_path: str, cols: Optional[list[str]], test: bool
    ) -> Iterator[pd.DataFrame]:
        for encoding in ["iso88591", "utf8", "latin1"]:
            for sep in [";", ","]:
                try:
                    # Try to read the entire CSV in chunks using a fixed encoding and delimiter
                    for i, chunk in enumerate(
                        pd.read_csv(
                            file_path,
                            sep=sep,
                            encoding=encoding,
                            chunksize=self.chunksize,
                            usecols=cols,
                        )
                    ):
                        if test and i >= 3:
                            break
                        yield chunk
                    # If we get here, reading succeeded, so break out of the encoding loop.
                    return
                except Exception as e:
                    self.logger.info(
                        f"Failed with encoding {encoding} and sep {sep}: {str(e)}"
                    )
                    continue
        # If none of the encoding/delimiter combinations work, raise an error.
        raise ValueError(
            f"Unable to read file {file_path} with any encoding and delimiter combination"
        )
