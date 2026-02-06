import os
import pickle
from typing import Dict

import pandas as pd
from tqdm import tqdm
from os.path import split
from ehr2meds.PREMEDS.preprocessing.constants import SUBJECT_ID
from ehr2meds.PREMEDS.preprocessing.io.data_handling import DataHandler
from ehr2meds.PREMEDS.preprocessing.premeds.concept_funcs import (
    factorize_subject_id,
    select_and_rename_columns,
)
from ehr2meds.PREMEDS.preprocessing.premeds.helpers import add_discharge_to_last_patient
from ehr2meds.PREMEDS.preprocessing.premeds.registers import RegisterConceptProcessor
from ehr2meds.PREMEDS.preprocessing.premeds.sp import ConceptProcessor
import logging

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
        
        # Get global datetime format
        datetime_format = cfg.get("datetime", {}).get("timeformat")

        # Create data handler for concepts
        self.data_handler = DataHandler(
            output_dir=cfg.paths.output,
            file_type=cfg.write_file_type,
            path=cfg.paths.concepts,
            chunksize=self.chunksize,
            test_rows=cfg.get("test_rows", 1_000_000),
            test=cfg.test,
            datetime_format=datetime_format,
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
                datetime_format=datetime_format,
            )

            # Create data handler for mappings
            self.link_file_handler = DataHandler(
                output_dir=cfg.paths.output,
                file_type=cfg.write_file_type,
                path=split(cfg.paths.pid_link)[0],
                chunksize=self.chunksize,  # not used here
                test_rows=cfg.get("test_rows", 1_000_000),
                test=cfg.test,
                datetime_format=datetime_format,
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
        Expects a 'files' list in config, where each entry is a dict with:
        - filename: name of the file
        - rename_columns: file-specific column renaming (required)

        Returns:
            Dict[str, int]: Mapping from original patient IDs to integer IDs
        """
        logger.info("Load patients info")
        
        # Get files list - always expect a list
        files = self.cfg.patients_info.files
        
        # Load all files and concatenate
        dfs = []
        for file_config in files:
            filename = file_config["filename"]
            # Each file must specify its own rename_columns
            rename_columns = file_config["rename_columns"]
            
            logger.info(f"Loading patient info from: {filename}")
            cols = list(rename_columns.keys()) if rename_columns else None
            df_file = self.data_handler.load_pandas(filename, cols=cols)
            
            # Use file-specific rename_columns
            df_file = select_and_rename_columns(df_file, rename_columns)
            dfs.append(df_file)
        
        # Concatenate all dataframes
        if len(dfs) > 1:
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Concatenated {len(dfs)} patient info files")
            # Drop duplicates based on subject_id if it exists
            if SUBJECT_ID in df.columns:
                initial_len = len(df)
                df = df.drop_duplicates(subset=[SUBJECT_ID], keep='first')
                if len(df) < initial_len:
                    logger.info(f"Removed {initial_len - len(df)} duplicate patient records")
        else:
            df = dfs[0]
        
        logger.info(f"Number of patients after selecting columns: {len(df)}")

        df, hash_to_int_map = factorize_subject_id(df)
        # Save the mapping for reference.
        with open(f"{self.cfg.paths.output}/hash_to_integer_map.pkl", "wb") as f:
            pickle.dump(hash_to_int_map, f)

        df = df.dropna(subset=[SUBJECT_ID], how="any")
        logger.info(f"Number of patients before saving: {len(df)}")
        self.data_handler.save(df, "subject")

        return hash_to_int_map

    def format_concepts(self, subject_id_mapping: Dict[str, int]) -> None:
        """Process all medical concepts"""
        # Get global datetime config if it exists
        global_datetime = self.cfg.get("datetime", {})
        
        for concept_type, concept_config in self.cfg.get("concepts", {}).items():
            # Merge global datetime config into concept config (concept config takes precedence)
            concept_config_with_global = concept_config.copy()
            concept_config_with_global["_global_datetime"] = global_datetime
            
            if concept_type == "admissions":
                self.format_admissions(concept_config_with_global, subject_id_mapping)
                continue  # continue to next concept
            
            # Check if this concept has multiple files (files list) or single file (filename)
            if "files" in concept_config:
                # Process multiple files for the same concept
                for file_config in concept_config["files"]:
                    # Create a merged config with file-specific settings
                    merged_config = concept_config_with_global.copy()
                    merged_config["filename"] = file_config["filename"]
                    if "rename_columns" in file_config:
                        merged_config["rename_columns"] = file_config["rename_columns"]
                    # File-specific configs override concept-level configs
                    for key, value in file_config.items():
                        if key not in ["filename", "rename_columns"]:
                            merged_config[key] = value
                    
                    try:
                        self._process_concept_chunks(
                            concept_type, merged_config, subject_id_mapping
                        )
                    except Exception as e:
                        logger.warning(f"Error processing {concept_type} from {file_config['filename']}: {str(e)}")
            else:
                # Single file (backward compatible)
                try:
                    self._process_concept_chunks(
                        concept_type, concept_config_with_global, subject_id_mapping
                    )
                except Exception as e:
                    logger.warning(f"Error processing {concept_type}: {str(e)}")

    def format_register_concepts(self, subject_id_mapping: Dict[str, int]) -> None:
        """Process the register concepts using the register-specific data handler"""
        # Load the register-SP mapping once - this maps register PIDs to SP hashes
        register_sp_link = self._get_register_sp_link()
        
        # Get global datetime config if it exists
        global_datetime = self.cfg.get("datetime", {})

        for concept_type, concept_config in self.cfg.get(
            "register_concepts", {}
        ).items():
            # Merge global datetime config into concept config (concept config takes precedence)
            concept_config_with_global = concept_config.copy()
            concept_config_with_global["_global_datetime"] = global_datetime
            
            logger.info(f"Processing register concept: {concept_type}")
            try:
                self.process_register_concept_chunks(
                    concept_type,
                    concept_config_with_global,
                    subject_id_mapping,
                    register_sp_link,
                )
            except Exception as e:
                logger.warning(f"Error processing {concept_type}: {str(e)}")

    def process_register_concept_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        register_sp_link: pd.DataFrame,
    ) -> None:
        chunk_idx = 0
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
                join_link_col=self.cfg.pid_link.join_col,  #  for linking to sp data
                target_link_col=self.cfg.pid_link.target_col,  #  for linking to sp data
            )

            self._safe_save(
                self.register_data_handler, processed_chunk, concept_type, chunk_idx, concept_config
            )
            chunk_idx += 1

    def _process_concept_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
    ) -> None:
        chunk_idx = 0
        for chunk in tqdm(
            self.data_handler.load_chunks(concept_config),
            desc=f"Chunks {concept_type}",
        ):
            processed_chunk = self.concept_processor.process(
                chunk, concept_config, subject_id_mapping
            )
            self._safe_save(
                self.data_handler, processed_chunk, concept_type, chunk_idx, concept_config
            )
            chunk_idx += 1

    def _safe_save(
        self, data_handler, processed_chunk, concept_type, chunk_idx: int, concept_config: dict
    ) -> None:
        if not processed_chunk.empty:
            # Check if save_in_chunks is set for this specific concept type
            save_in_chunks = concept_config.get("save_in_chunks", False)
            
            if save_in_chunks:
                # Save each chunk as a separate file in a directory named after concept_type
                # Create directory if it doesn't exist
                chunk_dir = os.path.join(data_handler.output_dir, concept_type)
                os.makedirs(chunk_dir, exist_ok=True)
                # Save file as concept_type/chunk_idx
                filename = os.path.join(concept_type, f"chunk_{chunk_idx}")
                data_handler.save(processed_chunk, filename, mode="w")
            else:
                # Original behavior: append to single file
                mode = "w" if chunk_idx == 0 else "a"
                data_handler.save(processed_chunk, concept_type, mode=mode)
        else:
            logger.warning(f"Empty processed chunk for {concept_type}, skipping save")

    def _get_register_sp_link(self) -> pd.DataFrame:
        pid_link_cfg = self.cfg.pid_link
        register_sp_link = self.link_file_handler.load_pandas(
            split(self.cfg.paths.pid_link)[1],
            cols=[pid_link_cfg.join_col, pid_link_cfg.target_col],
        )
        return register_sp_link

    def format_admissions(
        self, admissions_config: dict, subject_id_mapping: Dict[str, int]
    ) -> None:
        """Process the admissions concept separately, handling patients across chunks."""
        chunk_idx = 0
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

            self._safe_save(
                self.data_handler, processed_chunk, "admissions", chunk_idx, admissions_config
            )
            chunk_idx += 1

        # Process any remaining last patient data
        final_df = add_discharge_to_last_patient(last_patient_data)
        if not final_df.empty:
            save_in_chunks = admissions_config.get("save_in_chunks", False)
            if save_in_chunks:
                # Save final patient data as last chunk in directory
                chunk_dir = os.path.join(self.data_handler.output_dir, "admissions")
                os.makedirs(chunk_dir, exist_ok=True)
                filename = os.path.join("admissions", f"chunk_{chunk_idx}")
                self.data_handler.save(final_df, filename, mode="w")
            else:
                self.data_handler.save(final_df, "admissions", mode="a")
