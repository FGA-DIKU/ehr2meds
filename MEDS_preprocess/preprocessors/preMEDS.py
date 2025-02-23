import os
import pickle
from datetime import timedelta
from typing import Dict

import pandas as pd
from azure_run import datastore
from azureml.core import Dataset
from tqdm import tqdm

CODE = "code"
SUBJECT_ID = "subject_id"
ADMISSION = "admission"
DISCHARGE = "discharge"
TIMESTAMP = "timestamp"
FILENAME = "filename"
MANDATORY_COLUMNS = [SUBJECT_ID, CODE, TIMESTAMP]


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

        self.datastore = datastore(cfg.data_path.concepts.datastore)
        self.dump_path = cfg.data_path.concepts.dump_path

    def __call__(self):
        subject_id_mapping = self.format_patients_info()
        self.format_concepts(subject_id_mapping)

    def format_patients_info(self) -> Dict[str, int]:
        """
        Load and process patient information, creating a mapping of patient IDs.

        Returns:
            Dict[str, int]: Mapping from original patient IDs to integer IDs
        """
        self.logger.info("Load patients info")
        df = self.load_pandas(self.cfg.patients_info)
        if self.test:
            df = df.sample(100000)

        # Use columns_map to subset and rename the columns.
        columns_map = self.cfg.patients_info.get("columns_map", {})
        self.check_columns(df, columns_map)
        df = df[list(columns_map.keys())]
        df = df.rename(columns=columns_map)

        self.logger.info(f"Number of patients after selecting columns: {len(df)}")

        # Factorize the subject_id column into an integer mapping.
        df["integer_id"], unique_vals = pd.factorize(df[SUBJECT_ID])
        hash_to_int_map = dict(zip(unique_vals, range(len(unique_vals))))

        # Save the mapping for reference.
        with open(f"{self.cfg.paths.output_dir}/hash_to_integer_map.pkl", "wb") as f:
            pickle.dump(hash_to_int_map, f)

        # Overwrite subject_id with the factorized integer and drop the helper column.
        df[SUBJECT_ID] = df["integer_id"]
        df.drop(columns=["integer_id"], inplace=True)

        df.dropna(subset=[SUBJECT_ID], how="any", inplace=True)
        self.logger.info(f"Number of patients before saving: {len(df)}")
        self.save(df, self.cfg.patients_info, "subject")

        return hash_to_int_map

    def format_concepts(self, subject_id_mapping: Dict[str, int]):
        """Loop over all top-level concepts and process them with a single pipeline."""
        for concept_type, concept_config in tqdm(
            self.cfg.concepts.items(), desc="Concepts"
        ):
            first_chunk = True

            # A concept might have multiple input files:
            filenames = concept_config.get(FILENAME)
            if isinstance(filenames, list):
                for file_name in filenames:
                    concept_config[FILENAME] = file_name
                    self.process_concept_in_chunks(
                        concept_type, concept_config, subject_id_mapping, first_chunk
                    )
                    first_chunk = False
            else:
                # Single file
                self.process_concept_in_chunks(
                    concept_type, concept_config, subject_id_mapping, first_chunk
                )

    def process_concept_in_chunks(
        self,
        concept_type: str,
        concept_config: dict,
        subject_id_mapping: Dict[str, int],
        first_chunk: bool,
    ):
        """
        Process a single concept's data in chunks to manage memory usage.

        Args:
            concept_type: Type of medical concept being processed
            concept_config: Configuration for processing this concept
            subject_id_mapping: Dictionary mapping original patient IDs to integer IDs
            first_chunk: Whether this is the first chunk being processed
        """
        for chunk in tqdm(
            self.load_chunks(concept_config), desc=f"Chunks {concept_type}"
        ):
            processed_chunk = self.generic_concept_pipeline(
                chunk, concept_config, subject_id_mapping
            )
            if first_chunk:
                self.save(processed_chunk, concept_config, concept_type, mode="w")
                first_chunk = False
            else:
                self.save(processed_chunk, concept_config, concept_type, mode="a")

    def generic_concept_pipeline(
        self, df: pd.DataFrame, concept_config: dict, subject_id_mapping: dict
    ) -> pd.DataFrame:
        """
        1) Keep only the columns specified in columns_map
        2) Rename columns based on columns_map
        3) Optionally extract code using a regex on the main code column
        4) Fill missing code values using fillna_column, optionally with a separate regex
        5) Add a code prefix if needed
        6) Convert numeric columns if specified, map subject_id, drop NaNs and duplicates, and postprocess if needed
        """
        # 1) Select only the columns provided in columns_map
        df = self._select_and_rename_columns(df, concept_config.get("columns_map", {}))

        df = self._process_codes(df, concept_config)
        df = self._convert_and_clean_data(df, concept_config, subject_id_mapping)

        # 9) Call any postprocessing function if specified (e.g., merging admissions)
        postprocess = concept_config.get("postprocess")
        if postprocess:
            df = self._postprocess_switch(df, postprocess)

        return df

    @staticmethod
    def _select_and_rename_columns(df: pd.DataFrame, columns_map: dict) -> pd.DataFrame:
        """Select and rename columns based on columns_map."""
        MEDSPreprocessor.check_columns(df, columns_map)
        df = df[list(columns_map.keys())]
        df = df.rename(columns=columns_map)
        return df

    @staticmethod
    def _process_codes(df: pd.DataFrame, concept_config: dict) -> pd.DataFrame:
        """Handle code extraction, filling missing values, and adding prefixes."""
        # Extract code with regex if requested
        regex = concept_config.get("code_extraction_regex")
        if regex and CODE in df.columns:
            df[CODE] = df[CODE].str.extract(regex, expand=False)

        # Fill missing values
        fillna_cfg = concept_config.get("fillna")
        if fillna_cfg:
            df = MEDSPreprocessor._fill_missing_values(df, fillna_cfg)

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
        if MANDATORY_COLUMNS in df.columns:
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
    def _postprocess_switch(
        df: pd.DataFrame, postprocess_func_name: str
    ) -> pd.DataFrame:
        """
        Route to the appropriate postprocessing function.
        Example: for admissions, we handle merges of overlapping intervals.
        """
        if postprocess_func_name == "merge_admissions":
            return MEDSPreprocessor._merge_admissions(df)
        else:
            # Potentially handle other postprocesses or raise an error
            raise ValueError(f"Unknown postprocess function: {postprocess_func_name}")

    @staticmethod
    def _merge_admissions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Example of special post-processing for admissions:
        - Sort by subject_id and admission start
        - Merge intervals that overlap or are within 24h
        - Then produce final rows
        """
        # Drop rows where admission or discharge is missing
        df.dropna(subset=[ADMISSION, DISCHARGE], inplace=True)
        df[ADMISSION] = pd.to_datetime(df[ADMISSION], errors="coerce")
        df[DISCHARGE] = pd.to_datetime(df[DISCHARGE], errors="coerce")

        df.sort_values(by=[SUBJECT_ID, ADMISSION], inplace=True)

        merged = []
        current = None

        for _, row in df.iterrows():
            if current is None:
                current = row
                continue
            # If next admission is within 24h
            if row[ADMISSION] <= current[DISCHARGE] + timedelta(hours=24):
                # Extend discharge if needed
                if row[DISCHARGE] > current[DISCHARGE]:
                    current[DISCHARGE] = row[DISCHARGE]
            else:
                # add the completed admission
                merged.append(current)
                current = row
        if current is not None:
            merged.append(current)

        final_df = pd.DataFrame(merged)
        # Possibly set "timestamp" = "admission" for a final column if you want
        final_df[TIMESTAMP] = final_df[ADMISSION]
        return final_df

    # --------------------------------------------------------------------
    # Utility: load data
    # --------------------------------------------------------------------
    def load_pandas(self, cfg: dict):
        ds = self.get_dataset(cfg)
        df = ds.to_pandas_dataframe()
        return df

    def load_chunks(self, cfg: dict):
        ds = self.get_dataset(cfg)
        chunk_size = cfg.get("chunksize", 100000)
        i = cfg.get("start_chunk", 0)

        while True:
            self.logger.info(f"chunk {i}")
            chunk = ds.skip(i * chunk_size).take(chunk_size)
            df = chunk.to_pandas_dataframe()
            if df.empty:
                self.logger.info("Reached empty chunk, done.")
                break
            i += 1
            yield df

    def get_dataset(self, cfg: dict):
        """Create a TabularDataset from the file path in config."""
        file_path = (
            os.path.join(self.dump_path, cfg[FILENAME])
            if self.dump_path is not None
            else cfg[FILENAME]
        )
        ds = Dataset.Tabular.from_parquet_files(path=(self.datastore, file_path))
        if self.test:
            ds = ds.take(100000)
        return ds

    # --------------------------------------------------------------------
    # Utility: save data
    # --------------------------------------------------------------------
    def save(self, df: pd.DataFrame, cfg: dict, filename: str, mode: str = "w"):
        """
        Save the processed data to a file.

        Args:
            df: DataFrame containing the processed data
            cfg: Configuration for saving the data
            filename: Name of the file to save
            mode: Mode for saving the file ("w" for write, "a" for append)
        """
        self.logger.info(f"Saving {filename}")
        out_dir = self.cfg.paths.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Decide on filetype
        file_type = cfg.get("file_type", self.cfg.paths.file_type)
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
