import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple, List

import pandas as pd
from tqdm import tqdm

from MEDS_preprocess.preprocessors.constants import (
    ADMISSION,
    CODE,
    DISCHARGE,
    FILENAME,
    MANDATORY_COLUMNS,
    SUBJECT_ID,
)
from MEDS_preprocess.preprocessors.azure_load import get_data_loader


@dataclass
class DataConfig:
    """Configuration for data handling"""

    output_dir: str
    file_type: str
    datastore: Optional[str] = None
    dump_path: Optional[str] = None


class ConceptProcessor:
    """Handles the processing of medical concepts"""

    @staticmethod
    def process_concept(
        df: pd.DataFrame, concept_config: dict, subject_id_mapping: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Main method for processing a single concept's data
        """
        df = ConceptProcessor._select_and_rename_columns(
            df, concept_config.get("columns_map", {})
        )
        df = ConceptProcessor._process_codes(df, concept_config)
        df = ConceptProcessor._convert_and_clean_data(
            df, concept_config, subject_id_mapping
        )

        return df

    @staticmethod
    def process_register_concept(
        df: pd.DataFrame,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        data_handler: "DataHandler",
        register_sp_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process the register concepts."""
        # Step 1: Initial processing
        df = ConceptProcessor._process_initial_register_data(df, concept_config)

        # Step 2: Apply secondary mapping if needed
        df = ConceptProcessor._apply_secondary_mapping(df, concept_config, data_handler)

        # Step 3: Convert numeric columns
        df = ConceptProcessor._convert_numeric_columns(df, concept_config)

        # Step 4: Apply main mapping and register mapping
        df = ConceptProcessor._apply_main_and_register_mapping(
            df, concept_config, data_handler, register_sp_mapping
        )

        # Step 5: Process codes (unroll or prefix)
        df = ConceptProcessor._process_register_codes(df, concept_config)

        # Step 6: Final cleanup and mapping
        df = ConceptProcessor._map_and_clean_data(df, subject_id_mapping)

        return df

    @staticmethod
    def _process_initial_register_data(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Handle initial data processing steps."""
        # Select and rename columns
        df = ConceptProcessor._select_and_rename_columns(
            df, concept_config.get("columns_map", {})
        )

        # Combine datetime columns if needed
        df = ConceptProcessor._combine_datetime_columns(df, concept_config)

        return df

    @staticmethod
    def _apply_secondary_mapping(
        df: pd.DataFrame, concept_config: dict, data_handler: "DataHandler"
    ) -> pd.DataFrame:
        """Apply secondary mapping (e.g., vnr to drug name)."""
        if "secondary_mapping" not in concept_config:
            return df

        mapping_cfg = concept_config["secondary_mapping"]
        mapping_df = ConceptProcessor._load_mapping_file(mapping_cfg, data_handler)

        code_col = mapping_cfg.get("code_column")
        if code_col:
            df = ConceptProcessor._apply_code_mapping(
                df, mapping_df, mapping_cfg, code_col
            )
        else:
            df = ConceptProcessor._apply_simple_mapping(df, mapping_df, mapping_cfg)

        return df

    @staticmethod
    def _apply_code_mapping(
        df: pd.DataFrame, mapping_df: pd.DataFrame, mapping_cfg: dict, code_col: str
    ) -> pd.DataFrame:
        """Apply mapping with specific code column handling."""
        df = pd.merge(
            df,
            mapping_df[[mapping_cfg.get("right_on"), code_col]],
            left_on=mapping_cfg.get("left_on"),
            right_on=mapping_cfg.get("right_on"),
            how="inner",
        )
        df[CODE] = df[code_col]
        df = df.drop(columns=[mapping_cfg.get("left_on"), code_col])
        return df

    @staticmethod
    def _apply_simple_mapping(
        df: pd.DataFrame, mapping_df: pd.DataFrame, mapping_cfg: dict
    ) -> pd.DataFrame:
        """Apply simple mapping without code column."""
        df = pd.merge(
            df,
            mapping_df,
            left_on=mapping_cfg.get("left_on"),
            right_on=mapping_cfg.get("right_on"),
            how="inner",
        )
        df = df.drop(columns=[mapping_cfg.get("left_on")])
        return df

    @staticmethod
    def _apply_main_and_register_mapping(
        df: pd.DataFrame,
        concept_config: dict,
        data_handler: "DataHandler",
        register_sp_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply main mapping and register mapping to get final subject IDs."""
        if "main_mapping" not in concept_config:
            return df

        mapping_cfg = concept_config["main_mapping"]
        mapping_df = ConceptProcessor._load_mapping_file(mapping_cfg, data_handler)

        # Apply main mapping
        df = pd.merge(
            df,
            mapping_df,
            left_on=mapping_cfg.get("left_on"),
            right_on=mapping_cfg.get("right_on"),
            how="inner",
        )

        # Apply register mapping if possible
        if "PID" in df.columns and "PID" in register_sp_mapping.columns:
            df = pd.merge(df, register_sp_mapping, on="PID", how="inner")
            if SUBJECT_ID in register_sp_mapping.columns:
                df = df.drop(columns=[mapping_cfg.get("left_on"), "PID"])
                df = df.rename(columns={"SP_HASH": SUBJECT_ID})
        else:
            df = df.drop(columns=[mapping_cfg.get("left_on")])

        return df

    @staticmethod
    def _process_register_codes(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
        """Process codes through unrolling or adding prefixes."""
        if "unroll_columns" in concept_config:
            processed_dfs = ConceptProcessor._unroll_columns(df, concept_config)
            return pd.concat(processed_dfs, ignore_index=True) if processed_dfs else df

        # Add code prefix if specified
        code_prefix = concept_config.get("code_prefix", "")
        if code_prefix and CODE in df.columns:
            df[CODE] = code_prefix + df[CODE].astype(str)

        return df

    @staticmethod
    def _combine_datetime_columns(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Combine date and time columns into datetime columns."""
        if "combine_datetime" in concept_config:
            for target_col, date_time_cols in concept_config[
                "combine_datetime"
            ].items():
                date_col = date_time_cols.get("date_col")
                time_col = date_time_cols.get("time_col")
                if date_col in df.columns and time_col in df.columns:
                    df[target_col] = pd.to_datetime(
                        df[date_col].astype(str) + " " + df[time_col].astype(str),
                        errors="coerce",
                    )
                    # Drop original columns if requested
                    if date_time_cols.get("drop_original", True):
                        df = df.drop(columns=[date_col, time_col])
        return df

    @staticmethod
    def _convert_numeric_columns(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Convert specified columns to numeric type."""
        numeric_cols = concept_config.get("numeric_columns", [])
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _unroll_columns(df: pd.DataFrame, concept_config: dict) -> List[pd.DataFrame]:
        """
        Unroll specified columns into separate dataframes with code format.

        Returns a list of dataframes, each representing an unrolled column.
        """
        processed_dfs = []

        # Required columns to keep in each unrolled dataframe
        required_cols = [SUBJECT_ID]
        if "timestamp" in df.columns:
            required_cols.append("timestamp")

        # Keep only the columns that exist in the dataframe
        required_cols = [col for col in required_cols if col in df.columns]

        # For each column to unroll, create a separate df with it as CODE
        for col_info in concept_config["unroll_columns"]:
            col_name = col_info.get("column")
            if col_name in df.columns:
                # Create a copy with just the required columns and the unroll column
                unroll_df = df[required_cols + [col_name]].copy()

                # Apply prefix if specified
                prefix = col_info.get("prefix", "")

                # Rename to CODE
                unroll_df = unroll_df.rename(columns={col_name: CODE})

                # Add prefix to codes
                if prefix:
                    unroll_df[CODE] = prefix + unroll_df[CODE].astype(str)

                processed_dfs.append(unroll_df)

        return processed_dfs

    @staticmethod
    def _map_and_clean_data(df: pd.DataFrame, subject_id_mapping: dict) -> pd.DataFrame:
        """Map from SP PIDs to integer subject_ids and clean the data."""
        # Map from SP PIDs to integer subject_ids
        if SUBJECT_ID in df.columns:
            df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
            # Drop rows where mapping failed (subject_id is NaN)
            df = df.dropna(subset=[SUBJECT_ID])
            # Convert subject_id to integer
            df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)

        # Clean data
        if all(col in df.columns for col in MANDATORY_COLUMNS):
            df = df.dropna(subset=MANDATORY_COLUMNS, how="any")

        # Remove duplicates
        df = df.drop_duplicates()

        return df

    @staticmethod
    def _load_mapping_file(
        mapping_cfg: dict, data_handler: "DataHandler"
    ) -> pd.DataFrame:
        """Load a mapping file based on configuration."""
        filename = mapping_cfg.get("filename")
        # If data_handler is provided, use it to load from datastore
        return data_handler.load_pandas({"filename": filename})

    @staticmethod
    def _select_and_rename_columns(df: pd.DataFrame, columns_map: dict) -> pd.DataFrame:
        """Select and rename columns based on columns_map."""
        ConceptProcessor.check_columns(df, columns_map)
        df = df[list(columns_map.keys())]
        df = df.rename(columns=columns_map)
        return df

    @staticmethod
    def _process_codes(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
        """Filling missing values, and adding prefixes."""
        # Fill missing values
        fillna_cfg = concept_config.get("fillna")
        if fillna_cfg:
            df = ConceptProcessor._fill_missing_values(df, fillna_cfg)

        # Add code prefix if configured
        code_prefix = concept_config.get("code_prefix", "")
        if code_prefix and CODE in df.columns:
            df[CODE] = code_prefix + df[CODE].astype(str)

        return df

    @staticmethod
    def _convert_and_clean_data(
        df: pd.DataFrame, concept_config: dict, subject_id_mapping: dict
    ) -> pd.DataFrame:
        """Convert numeric columns, map subject IDs, and clean the data."""
        # Convert numeric columns
        numeric_cols = concept_config.get("numeric_columns", [])
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Map subject_id if available
        if SUBJECT_ID in df.columns:
            df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
            # Drop rows where mapping failed
            df = df.dropna(subset=[SUBJECT_ID])
            # Convert to integer
            df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)

        # Clean data
        if all(col in df.columns for col in MANDATORY_COLUMNS):
            df = df.dropna(subset=MANDATORY_COLUMNS, how="any")
        df = df.drop_duplicates()

        return df

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame, fillna_cfg: dict) -> pd.DataFrame:
        """
        Fill missing values using specified columns and regex patterns.
        Drop the columns used to fill missing values.
        """
        for target_col, fill_config in fillna_cfg.items():
            fill_col = fill_config.get("column")
            if fill_col and fill_col in df.columns:
                fillna_regex = fill_config.get("regex")
                if fillna_regex:
                    fill_vals = df[fill_col].str.extract(fillna_regex, expand=False)
                else:
                    fill_vals = df[fill_col]
                df[target_col] = df[target_col].fillna(fill_vals)
                df = df.drop(columns=[fill_col])
        return df

    @staticmethod
    def process_adt_admissions(
        df: pd.DataFrame, 
        admissions_config: dict, 
        subject_id_mapping: Dict[str, int],
        last_patient_data: Optional[dict] = None
    ) -> Tuple[pd.DataFrame, Optional[dict]]:
        """
        Process ADT admissions data to create admission/discharge events and department transfers.
        Handles patients that span across chunks.
        
        Args:
            df: Input DataFrame
            admissions_config: Configuration for admissions processing
            subject_id_mapping: Mapping from original IDs to integer IDs
            last_patient_data: Data from the last patient in previous chunk
        
        Returns:
            Tuple containing:
            - Processed DataFrame with events
            - Data for the last patient if it's incomplete (spans to next chunk)
        """
        # First select and rename columns
        df = ConceptProcessor._select_and_rename_columns(
            df, admissions_config.get("columns_map", {})
        )
        # Map subject_id
        if SUBJECT_ID in df.columns:
            df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
            df = df.dropna(subset=[SUBJECT_ID])
            df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)

        # Sort by patient and timestamp
        df = df.sort_values([SUBJECT_ID, "timestamp_in"])
        
        # Initialize lists to store the processed events
        events = []
        
        # If we have data from last chunk's patient, initialize with that
        current_patient_id = None
        admission_start = None
        last_transfer = None
        
        if last_patient_data:
            current_patient_id = last_patient_data["subject_id"]
            admission_start = last_patient_data["admission_start"]
            last_transfer = last_patient_data["last_transfer"]
        
        # Process each patient's events
        for subject_id, patient_df in df.groupby(SUBJECT_ID):
            # If this is a new patient and we had a previous patient's data,
            # finalize the previous patient's events
            if current_patient_id is not None and subject_id != current_patient_id:
                if admission_start is not None and last_transfer is not None:
                    events.append({
                        SUBJECT_ID: current_patient_id,
                        "timestamp": last_transfer["timestamp_out"],
                        CODE: "DISCHARGE_ADT"
                    })
                
                # Reset for new patient
                admission_start = None
                last_transfer = None
            
            current_patient_id = subject_id
            for _, row in patient_df.iterrows():
                event_type = row["type"]
                dept = row["section"]
                timestamp_in = row["timestamp_in"]

                if event_type.lower() == "indlaeggelse":
                    # If there was a previous admission, add discharge at last transfer
                    if admission_start is not None and last_transfer is not None:
                        events.append({
                            SUBJECT_ID: subject_id,
                            "timestamp": last_transfer["timestamp_out"],
                            CODE: "DISCHARGE_ADT"
                        })
                    
                    # Start new admission
                    admission_start = row
                    events.append({
                        SUBJECT_ID: subject_id,
                        "timestamp": timestamp_in,
                        CODE: "ADMISSION_ADT"
                    })
                    
                    # Add department code
                    events.append({
                        SUBJECT_ID: subject_id,
                        "timestamp": timestamp_in,
                        CODE: f"AFSNIT_ADT_{dept}"
                    })
                    
                elif event_type.lower() == "flyt ind" and admission_start is not None:
                    # Record transfer
                    if admissions_config.get("save_adm_move", True):
                        events.append({
                            SUBJECT_ID: subject_id,
                            "timestamp": timestamp_in,
                            CODE: "ADM_move"
                        })
                    # Add department code for the new department
                    events.append({
                        SUBJECT_ID: subject_id,
                        "timestamp": timestamp_in,
                        CODE: f"AFSNIT_ADT_{dept}"
                    })
                    last_transfer = row
        
        # Convert events to DataFrame
        result_df = pd.DataFrame(events)
        # Sort by patient and timestamp
        if not result_df.empty:
            result_df = result_df.sort_values([SUBJECT_ID, "timestamp"])
        
        # Prepare data for the last patient if their events might continue in next chunk
        last_patient_info = None
        if current_patient_id is not None and admission_start is not None:
            last_patient_info = {
                "subject_id": current_patient_id,
                "admission_start": admission_start,
                "last_transfer": last_transfer,
                "events": []  # Will be populated if this is the final chunk
            }
        
        return result_df, last_patient_info

    @staticmethod
    def check_columns(df: pd.DataFrame, columns_map: dict):
        """Check if all columns in columns_map are present in df."""
        missing_columns = set(columns_map.keys()) - set(df.columns)
        if missing_columns:
            available_columns = pd.DataFrame({"Available Columns": sorted(df.columns)})
            requested_columns = pd.DataFrame(
                {"Requested Columns": sorted(columns_map.keys())}
            )
            error_msg = f"\nMissing columns: {sorted(missing_columns)}\n\n"
            error_msg += "Columns comparison:\n"
            error_msg += f"{pd.concat([available_columns, requested_columns], axis=1).to_string()}"
            raise ValueError(error_msg)


class MEDSPreprocessor:
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

        # Create data handler for concepts
        self.data_handler = DataHandler(
            DataConfig(
                output_dir=cfg.paths.output_dir,
                file_type=cfg.paths.file_type,
                datastore=cfg.data_path.concepts.get("datastore"),
                dump_path=cfg.data_path.concepts.dump_path,
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
                ),
                logger,
                env=cfg.env,
                test=self.test,
            )

        # Create data handler for mappings
        self.mapping_data_handler = DataHandler(
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
        df = self.data_handler.load_pandas(self.cfg.patients_info)

        # Use columns_map to subset and rename the columns.
        df = ConceptProcessor._select_and_rename_columns(
            df, self.cfg.patients_info.get("columns_map", {})
        )

        self.logger.info(f"Number of patients after selecting columns: {len(df)}")

        df, hash_to_int_map = self._factorize_subject_id(df)
        # Save the mapping for reference.
        with open(f"{self.cfg.paths.output_dir}/hash_to_integer_map.pkl", "wb") as f:
            pickle.dump(hash_to_int_map, f)

        df = df.dropna(subset=[SUBJECT_ID], how="any")
        self.logger.info(f"Number of patients before saving: {len(df)}")
        self.data_handler.save(df, "subject")

        return hash_to_int_map

    @staticmethod
    def _factorize_subject_id(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Factorize the subject_id column into an integer mapping."""
        df["integer_id"], unique_vals = pd.factorize(df[SUBJECT_ID])
        shifted_indices = df["integer_id"] + 1  # +1 to avoid binary dtype
        hash_to_int_map = dict(zip(unique_vals, shifted_indices))
        # Overwrite subject_id with the factorized integer and drop the helper column.
        df[SUBJECT_ID] = shifted_indices
        df = df.drop(columns=["integer_id"])
        return df, hash_to_int_map

    def format_concepts(self, subject_id_mapping: Dict[str, int]) -> None:
        """Process all medical concepts"""
        for concept_type, concept_config in tqdm(
            self.cfg.concepts.items(), desc="Concepts"
        ):
            if concept_type == "admissions":
                self.format_admissions(concept_config, subject_id_mapping)
                continue
            self._process_concept_chunks(
                concept_type, concept_config, subject_id_mapping, first_chunk=True
            )

    def format_register_concepts(self, subject_id_mapping: Dict[str, int]) -> None:
        """Process the register concepts using the register-specific data handler"""
        try:
            # Load the register-SP mapping once - this maps register PIDs to SP hashes
            register_sp_mapping = self.mapping_data_handler.load_pandas(
                {"filename": "register_sp_mapping.parquet"}
            )
            self.logger.info(
                f"Loaded register-SP mapping with {len(register_sp_mapping)} rows"
            )
        except Exception as e:
            self.logger.error(f"Failed to load register-SP mapping: {e}")
            self.logger.warning("Processing register concepts without SP mapping!")
            register_sp_mapping = pd.DataFrame(columns=["PID", "SP_HASH"])

        for concept_type, concept_config in tqdm(
            self.cfg.register_concepts.items(), desc="Register concepts"
        ):
            self.logger.info(f"Processing register concept: {concept_type}")
            first_chunk = True

            try:
                for chunk in tqdm(
                    self.register_data_handler.load_chunks(concept_config),
                    desc=f"Chunks {concept_type}",
                ):
                    processed_chunk = self.concept_processor.process_register_concept(
                        chunk,
                        concept_config,
                        subject_id_mapping,
                        self.register_data_handler,
                        register_sp_mapping,
                    )

                    # Only save if we have data
                    if not processed_chunk.empty:
                        mode = "w" if first_chunk else "a"
                        self.data_handler.save(processed_chunk, concept_type, mode=mode)
                        first_chunk = False
                    else:
                        self.logger.warning(
                            f"Empty processed chunk for {concept_type}, skipping save"
                        )
            except Exception as e:
                self.logger.error(f"Error processing {concept_type}: {e}")
                continue

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
            processed_chunk, last_patient_data = ConceptProcessor.process_adt_admissions(
                chunk, admissions_config, subject_id_mapping, last_patient_data
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
            processed_chunk = self.concept_processor.process_concept(
                chunk, concept_config, subject_id_mapping
            )
            mode = "w" if first_chunk else "a"
            self.data_handler.save(processed_chunk, concept_type, mode=mode)
            first_chunk = False


class DataHandler:
    """Handles data loading and saving operations"""

    def __init__(self, config: DataConfig, logger, env: str, test: bool):
        self.output_dir = config.output_dir
        self.file_type = config.file_type
        self.env = env
        self.logger = logger
        self.test = test

        # Initialize the appropriate data loader
        self.data_loader = get_data_loader(
            env=self.env,
            datastore=config.datastore,
            dump_path=config.dump_path,
            logger=logger,
        )

    def load_pandas(self, cfg: dict) -> pd.DataFrame:
        return self.data_loader.load_dataframe(
            filename=cfg[FILENAME], test=self.test, n_rows=1_000_000
        )

    def load_chunks(self, cfg: dict) -> Iterator[pd.DataFrame]:
        chunk_size = cfg.get("chunksize", 500_000)
        return self.data_loader.load_chunks(
            filename=cfg[FILENAME], chunk_size=chunk_size, test=self.test
        )

    def save(self, df: pd.DataFrame, filename: str, mode: str = "w") -> None:
        """
        Save the processed data to a file.

        Args:
            df: DataFrame containing the processed data
            filename: Name of the file to save
            mode: Mode for saving the file ("w" for write, "a" for append)
        """
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {filename}, skipping save")
            return

        self.logger.info(f"Saving {filename} with {len(df)} rows")
        out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Decide on filetype
        file_type = self.file_type
        path = os.path.join(out_dir, f"{filename}.{file_type}")

        if file_type == "parquet":
            if mode == "w" or not os.path.exists(path):
                df.to_parquet(path, index=False)
            else:
                # For append mode with parquet, we need to read, concat, then write
                existing_df = pd.read_parquet(path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_parquet(path, index=False)
        elif file_type == "csv":
            if mode == "w":
                df.to_csv(path, index=False, mode="w")
            else:
                # append without header
                df.to_csv(path, index=False, mode="a", header=False)
        else:
            raise ValueError(f"Filetype {file_type} not implemented.")
