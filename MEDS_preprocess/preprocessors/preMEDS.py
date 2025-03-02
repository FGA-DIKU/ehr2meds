import os
import pickle
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterator, Optional, Tuple, List

import pandas as pd
from azureml.core import Dataset
from MEDS_preprocess.azure_run import datastore
from tqdm import tqdm

from MEDS_preprocess.preprocessors.constants import (
    ADMISSION,
    CODE,
    DISCHARGE,
    FILENAME,
    MANDATORY_COLUMNS,
    SUBJECT_ID,
)


@dataclass
class DataConfig:
    """Configuration for data handling"""

    datastore: str
    dump_path: Optional[str]
    output_dir: str
    file_type: str


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

    def process_register_concept(
        self, df: pd.DataFrame, concept_config: dict, subject_id_mapping: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Process the register concepts.

        This method orchestrates the entire register concept processing workflow by calling
        specialized helper methods for each step.

        Args:
            df: The raw register dataframe
            concept_config: Configuration for processing
            subject_id_mapping: Mapping from original PIDs to integer subject_ids

        Returns:
            Processed dataframe with standardized columns
        """
        # Step 1: Select and rename columns
        df = self._select_and_rename_columns(df, concept_config.get("columns_map", {}))

        # Step 2: Combine date and time columns if needed
        df = self._combine_datetime_columns(df, concept_config)

        # Step 3: Apply secondary mapping if needed
        df = self._apply_secondary_mapping(df, concept_config)

        # Step 4: Convert numeric columns
        df = self._convert_numeric_columns(df, concept_config)

        # Step 5: Unroll columns or apply code prefix
        processed_dfs = self._unroll_columns(df, concept_config)

        # Step 6: Concatenate all processed dataframes
        result_df = (
            pd.concat(processed_dfs, ignore_index=True)
            if len(processed_dfs) > 0
            else df
        )

        # Step 7: Apply main mapping to get SP PIDs
        result_df = self._apply_main_mapping(result_df, concept_config)

        # Step 8: Map to integer subject IDs and clean data
        result_df = self._map_and_clean_data(result_df, subject_id_mapping)

        return result_df

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
    def _apply_secondary_mapping(
        df: pd.DataFrame, concept_config: dict
    ) -> pd.DataFrame:
        """Apply secondary mapping to map records to PIDs if needed."""
        if "secondary_mapping" in concept_config:
            mapping_cfg = concept_config["secondary_mapping"]
            mapping_df = ConceptProcessor._load_mapping_file(mapping_cfg)
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
        If no unrolling is configured, returns a list containing the original dataframe
        with code prefix applied (if configured).
        """
        processed_dfs = []

        # If we have columns to unroll
        if "unroll_columns" in concept_config:
            required_cols = [SUBJECT_ID]
            if "timestamp" in concept_config.get("columns_map", {}).values():
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
        else:
            # If no unrolling, just add code prefix if specified
            code_prefix = concept_config.get("code_prefix", "")
            if code_prefix and CODE in df.columns:
                df[CODE] = code_prefix + df[CODE].astype(str)
            processed_dfs.append(df)

        return processed_dfs

    @staticmethod
    def _apply_main_mapping(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
        """Map to SP PIDs via main mapping if needed."""
        if "main_mapping" in concept_config:
            mapping_cfg = concept_config["main_mapping"]
            mapping_df = ConceptProcessor._load_mapping_file(mapping_cfg)
            df = pd.merge(
                df,
                mapping_df,
                left_on=mapping_cfg.get("left_on"),
                right_on=mapping_cfg.get("right_on"),
                how="inner",
            )
            # Use the mapped SUBJECT_ID and drop the original
            df = df.rename(columns={mapping_cfg.get("right_on"): SUBJECT_ID})
            df = df.drop(columns=[mapping_cfg.get("left_on")])
        return df

    @staticmethod
    def _map_and_clean_data(df: pd.DataFrame, subject_id_mapping: dict) -> pd.DataFrame:
        """Map from SP PIDs to integer subject_ids and clean the data."""
        # Map from SP PIDs to integer subject_ids
        if SUBJECT_ID in df.columns:
            df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)

        # Clean data
        if all(col in df.columns for col in MANDATORY_COLUMNS):
            df.dropna(subset=MANDATORY_COLUMNS, how="any", inplace=True)
        df.drop_duplicates(inplace=True)

        return df

    @staticmethod
    def _load_mapping_file(mapping_cfg: dict) -> pd.DataFrame:
        """Load a mapping file based on configuration."""
        filename = mapping_cfg.get("filename")
        return pd.read_csv(filename)

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
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Map subject_id if available
        if SUBJECT_ID in df.columns:
            df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)

        # Clean data
        if all(col in df.columns for col in MANDATORY_COLUMNS):
            df.dropna(subset=MANDATORY_COLUMNS, how="any", inplace=True)
        df.drop_duplicates(inplace=True)

        return df

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame, fillna_cfg: dict):
        """
        Fill missing values using specified columns and regex patterns.
        Drop the columns used to fill missing values.
        """
        for target_col, fill_config in fillna_cfg.items():
            fill_col = fill_config.get("column")
            if fill_col:
                fillna_regex = fill_config.get("regex")
                if fillna_regex:
                    fill_vals = df[fill_col].str.extract(fillna_regex, expand=False)
                else:
                    fill_vals = df[fill_col]
                df[target_col] = df[target_col].fillna(fill_vals)
                df = df.drop(columns=[fill_col])
        return df

    @staticmethod
    def process_admissions(
        df: pd.DataFrame, admissions_config: dict, subject_id_mapping: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Process admissions data.
        Expected final columns: subject_id, admission, discharge (and optionally timestamp).
        """
        # For admissions, we only select & rename columns.
        df = ConceptProcessor._select_and_rename_columns(
            df, admissions_config.get("columns_map", {})
        )
        # For admissions, we do not process codes or convert numeric columns.
        # But we still allow a postprocessing step (e.g., merging overlapping intervals).
        df = ConceptProcessor._merge_admissions(df)
        # Map subject_id if available
        if SUBJECT_ID in df.columns:
            df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)

        # Clean data
        df.dropna(subset=[ADMISSION, DISCHARGE], how="any", inplace=True)
        df.drop_duplicates(inplace=True)
        return df

    @staticmethod
    def _merge_admissions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge overlapping admission intervals for each subject:
        - Group by subject_id
        - Sort by admission start time
        - Merge intervals that overlap or are within 24h
        """
        # Drop rows where admission or discharge is missing
        df.dropna(subset=[ADMISSION, DISCHARGE], inplace=True)
        df[ADMISSION] = pd.to_datetime(df[ADMISSION], errors="coerce")
        df[DISCHARGE] = pd.to_datetime(df[DISCHARGE], errors="coerce")

        # Sort by subject_id and admission time
        df = df.sort_values(by=[SUBJECT_ID, ADMISSION])

        merged = []

        # Process each subject separately
        for _, subject_df in df.groupby(SUBJECT_ID):
            current = None

            for _, row in subject_df.iterrows():
                if current is None:
                    current = row.copy()
                    continue

                # If next admission is within 24h of current discharge
                if row[ADMISSION] <= current[DISCHARGE] + timedelta(hours=24):
                    # Extend discharge if needed
                    if row[DISCHARGE] > current[DISCHARGE]:
                        current[DISCHARGE] = row[DISCHARGE]
                else:
                    # Add the completed admission
                    merged.append(current)
                    current = row.copy()

            if current is not None:
                merged.append(current)

        return pd.DataFrame(merged)

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

        # this will be simplified in the future
        data_config = DataConfig(
            datastore=cfg.data_path.concepts.datastore,
            dump_path=cfg.data_path.concepts.dump_path,
            output_dir=cfg.paths.output_dir,
            file_type=cfg.paths.file_type,
        )
        self.data_handler = DataHandler(data_config, logger)
        self.concept_processor = ConceptProcessor()

    def __call__(self):
        subject_id_mapping = self.format_patients_info()
        self.format_concepts(subject_id_mapping)
        self.format_register_concepts(subject_id_mapping)

    def format_patients_info(self) -> Dict[str, int]:
        """
        Load and process patient information, creating a mapping of patient IDs.

        Returns:
            Dict[str, int]: Mapping from original patient IDs to integer IDs
        """
        self.logger.info("Load patients info")
        df = self.data_handler.load_pandas(self.cfg.patients_info, self.test)

        # Use columns_map to subset and rename the columns.
        df = ConceptProcessor._select_and_rename_columns(
            df, self.cfg.patients_info.get("columns_map", {})
        )

        self.logger.info(f"Number of patients after selecting columns: {len(df)}")

        df, hash_to_int_map = self._factorize_subject_id(df)
        # Save the mapping for reference.
        with open(f"{self.cfg.paths.output_dir}/hash_to_integer_map.pkl", "wb") as f:
            pickle.dump(hash_to_int_map, f)

        df.dropna(subset=[SUBJECT_ID], how="any", inplace=True)
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
        df[SUBJECT_ID] = df["integer_id"]
        df.drop(columns=["integer_id"], inplace=True)
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
        """Process the register concepts"""
        for concept_type, concept_config in tqdm(
            self.cfg.register_concepts.items(), desc="Register concepts"
        ):
            self._process_register_concept_chunks(
                concept_type, concept_config, subject_id_mapping, first_chunk=True
            )

    def format_admissions(
        self, admissions_config: dict, subject_id_mapping: Dict[str, int]
    ) -> None:
        """Process the admissions concept separately."""
        first_chunk = True
        for chunk in tqdm(
            self.data_handler.load_chunks(admissions_config, self.test),
            desc="Chunks admissions",
        ):
            processed_chunk = ConceptProcessor.process_admissions(
                chunk, admissions_config, subject_id_mapping
            )
            mode = "w" if first_chunk else "a"
            self.data_handler.save(processed_chunk, "admissions", mode=mode)
            first_chunk = False

    def _process_concept_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        first_chunk: bool,
    ) -> None:
        for chunk in tqdm(
            self.data_handler.load_chunks(concept_config, self.test),
            desc=f"Chunks {concept_type}",
        ):
            processed_chunk = self.concept_processor.process_concept(
                chunk, concept_config, subject_id_mapping
            )
            mode = "w" if first_chunk else "a"
            self.data_handler.save(processed_chunk, concept_type, mode=mode)
            first_chunk = False

    def _process_register_concept_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        first_chunk: bool,
    ) -> None:
        for chunk in tqdm(
            self.data_handler.load_chunks(concept_config, self.test),
            desc=f"Chunks {concept_type}",
        ):
            processed_chunk = self.concept_processor.process_register_concept(
                chunk, concept_config, subject_id_mapping
            )
            mode = "w" if first_chunk else "a"
            self.data_handler.save(processed_chunk, concept_type, mode=mode)
            first_chunk = False


class DataHandler:
    """Handles data loading and saving operations"""

    def __init__(self, config: DataConfig, logger):
        self.datastore = datastore(config.datastore)
        self.dump_path = config.dump_path
        self.output_dir = config.output_dir
        self.file_type = config.file_type
        self.logger = logger

    def load_pandas(self, cfg: dict, test: bool = False) -> pd.DataFrame:
        ds: Dataset.Tabular = self._get_dataset(cfg[FILENAME])
        if test:
            ds = ds.take(500_000)
        return ds.to_pandas_dataframe()

    def load_chunks(self, cfg: dict, test: bool = False) -> Iterator[pd.DataFrame]:
        ds = self._get_dataset(cfg[FILENAME])
        chunk_size = cfg.get("chunksize", 500_000)
        i = cfg.get("start_chunk", 0)

        # Add a safety limit for test mode
        max_chunks = 2 if test else float("inf")
        chunks_processed = 0

        while chunks_processed < max_chunks:
            self.logger.info(f"chunk {i}")
            chunk = ds.skip(i * chunk_size).take(chunk_size)
            df = chunk.to_pandas_dataframe()
            if df.empty:
                self.logger.info("Reached empty chunk, done.")
                break
            i += 1
            chunks_processed += 1
            yield df

    def _get_dataset(self, filename: str) -> Dataset.Tabular:
        file_path = (
            os.path.join(self.dump_path, filename)
            if self.dump_path is not None
            else filename
        )
        ds = Dataset.Tabular.from_parquet_files(path=(self.datastore, file_path))
        return ds

    def save(self, df: pd.DataFrame, filename: str, mode: str = "w") -> None:
        """
        Save the processed data to a file.

        Args:
            df: DataFrame containing the processed data
            filename: Name of the file to save
            mode: Mode for saving the file ("w" for write, "a" for append)
        """
        self.logger.info(f"Saving {filename}")
        out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Decide on filetype
        file_type = self.file_type
        path = os.path.join(out_dir, f"{filename}.{file_type}")

        if file_type == "parquet":
            df.to_parquet(path)
        elif file_type == "csv":
            if mode == "w":
                df.to_csv(path, index=False, mode="w")
            else:
                # append without header
                df.to_csv(path, index=False, mode="a", header=False)
        else:
            raise ValueError(f"Filetype {file_type} not implemented.")
