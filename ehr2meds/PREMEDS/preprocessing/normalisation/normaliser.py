import logging
import os
import re
from os.path import dirname, isdir, join, split, splitext
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from ehr2meds.PREMEDS.preprocessing.io.dataloader import get_data_loader
from ehr2meds.PREMEDS.preprocessing.constants import CODE, SUBJECT_ID

logger = logging.getLogger(__name__)


class Normaliser:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.test = cfg.test
        logger.info(f"test {self.test}")
        self.normalisation_type = cfg.data["norm_type"]
        self._init_data_loader()

        # Initialize distribution data placeholders
        self.min_max_vals = None
        self.quantiles = None
        self.n_quantiles = None

        self.numeric_value = cfg.data["numeric_value"]

    def _init_data_loader(self):
        # Check if input is a directory (chunked files) or a single file
        self.input_is_directory = isdir(self.cfg.paths.input)
        
        if self.input_is_directory:
            # For directory input, we'll load chunks directly from files
            # No need for a data loader with chunksize
            self.data_loader = get_data_loader(
                path=self.cfg.paths.input,
                env=self.cfg.env,
                chunksize=None,  # Not used for directory loading
                test=self.test,
                test_rows=self.cfg.data.get("test_rows", 100_000),
            )
        else:
            # For single file input, use existing behavior
            self.data_loader = get_data_loader(
                path=dirname(self.cfg.paths.input),
                env=self.cfg.env,
                chunksize=self.cfg.data.chunksize,
                test=self.test,
                test_rows=self.cfg.data.get("test_rows", 100_000),
            )

    def __call__(self):
        print("Getting lab distribution")
        dist = self.get_lab_values()
        self._process_distribution_data(dist)
        print("Normalising data")
        self.normalise_data()

    def normalise_data(self):
        counter = 0
        for chunk in tqdm(
            self._get_chunks(),
            desc="Processing chunks",
        ):
            chunk = self._prepare_chunk(chunk, counter)
            self._save_chunk(chunk, counter)

            counter += 1

    def _process_distribution_data(self, dist: Dict[str, List[float]]) -> None:
        """Process distribution data based on normalization type."""
        if self.normalisation_type == "Min_max":
            self._process_minmax_distribution(dist)
        elif self.normalisation_type == "Categorise":
            self._process_category_distribution(dist)
        elif self.normalisation_type == "Quantiles":
            self._process_quantile_distribution(dist)
        else:
            raise ValueError("Invalid type of normalisation")

    def _process_minmax_distribution(self, dist: Dict[str, List[float]]) -> None:
        """Process distribution data for min-max normalization."""
        self.min_max_vals = {
            concept: (
                (
                    np.percentile(dist[concept], 0.01 * 100)
                    if len(dist[concept]) > 1
                    else dist[concept][0]
                ),
                (
                    np.percentile(dist[concept], 0.99 * 100)
                    if len(dist[concept]) > 1
                    else dist[concept][0]
                ),
            )
            for concept in dist
            if dist[concept]
        }

    def _process_category_distribution(self, dist: Dict[str, List[float]]) -> None:
        """Process distribution data for categorization."""
        self.quantiles = {
            concept: (
                np.percentile(sorted(dist[concept]), [25, 50, 75])
                if len(dist[concept]) > 0
                else (0, 0, 0)
            )
            for concept in dist
        }

    def _process_quantile_distribution(self, dist: Dict[str, List[float]]) -> None:
        """Process distribution data for quantile normalization."""
        self.n_quantiles = self.cfg.data["n_quantiles"]
        self.quantiles = {
            concept: (
                [
                    np.percentile(sorted(dist[concept]), i)
                    for i in np.linspace(100 / self.n_quantiles, 100, self.n_quantiles)
                ]
                if len(dist[concept]) > 0
                else [0] * self.n_quantiles
            )
            for concept in dist
        }

    def _prepare_chunk(self, chunk: pd.DataFrame, counter: int) -> pd.DataFrame:
        """Prepare and process a single chunk of data."""
        # Ensure subject_id is treated as integer
        chunk[SUBJECT_ID] = pd.to_numeric(chunk[SUBJECT_ID]).astype("Int64")
        chunk = chunk.reset_index(drop=True)
        logger.info(f"Loaded chunk {counter}")
        return self.process_chunk(chunk)

    def _save_chunk(self, chunk: pd.DataFrame, counter: int) -> None:
        """Save a processed chunk to file."""
        if self.input_is_directory:
            # If input was chunked, save output as chunks too
            base_name, ext = splitext(self.cfg.file_name)
            # Create directory if it doesn't exist
            chunk_dir = join(self.cfg.paths.output_dir, base_name)
            os.makedirs(chunk_dir, exist_ok=True)
            # Save file as base_name/chunk_counter.ext
            chunk_filename = f"chunk_{counter}{ext}"
            save_path = join(chunk_dir, chunk_filename)
            mode = "w"
            header = True  # Always write header for chunked files
        else:
            # Original behavior: append to single file
            save_path = join(self.cfg.paths.output_dir, self.cfg.file_name)
            mode = "w" if counter == 0 else "a"
            header = (counter == 0)  # Write header only for first chunk
        
        if self.cfg.file_name.endswith(".parquet"):
            chunk.to_parquet(save_path, index=False, mode=mode)
        else:
            chunk.to_csv(save_path, index=False, mode=mode, header=header)

    def _get_chunks(self) -> Iterator[pd.DataFrame]:
        """Get chunks from either a directory of chunk files or a single file."""
        if self.input_is_directory:
            yield from self._load_chunks_from_directory()
        else:
            yield from self.data_loader.load_chunks(
                filename=split(self.cfg.paths.input)[1],
            )

    def _load_chunks_from_directory(self) -> Iterator[pd.DataFrame]:
        """Load chunks from a directory containing chunk files (chunk_0.csv, chunk_1.csv, etc.)."""
        input_dir = self.cfg.paths.input
        
        # Get all files in the directory
        all_files = os.listdir(input_dir)
        
        # Filter for chunk files (chunk_0.csv, chunk_1.csv, etc.)
        chunk_files = [
            f for f in all_files 
            if re.match(r'chunk_\d+\.(csv|parquet)$', f)
        ]
        
        if not chunk_files:
            raise ValueError(
                f"No chunk files found in directory {input_dir}. "
                f"Expected files matching pattern 'chunk_*.csv' or 'chunk_*.parquet'"
            )
        
        # Sort by chunk number
        def get_chunk_number(filename):
            match = re.search(r'chunk_(\d+)', filename)
            return int(match.group(1)) if match else -1
        
        chunk_files.sort(key=get_chunk_number)
        
        # Limit chunks in test mode
        max_chunks = 2 if self.test else len(chunk_files)
        
        # Load each chunk file
        for chunk_file in chunk_files[:max_chunks]:
            chunk_path = join(input_dir, chunk_file)
            if chunk_file.endswith(".parquet"):
                yield pd.read_parquet(chunk_path)
            else:
                # Try different encodings and separators
                loaded = False
                for encoding in ["utf8", "iso88591", "latin1"]:
                    for sep in [",", ";"]:
                        try:
                            df = pd.read_csv(chunk_path, sep=sep, encoding=encoding)
                            yield df
                            loaded = True
                            break
                        except Exception:
                            continue
                    if loaded:
                        break
                if not loaded:
                    raise ValueError(f"Unable to read chunk file {chunk_path}")

    def get_lab_values(self):
        logger.info("Getting lab distribution")
        lab_val_dict = {}
        counter = 0

        for chunk in tqdm(
            self._get_chunks(),
            desc="Building lab distribution",
        ):
            if self.numeric_value not in chunk.columns or CODE not in chunk.columns:
                raise ValueError(
                    f"Missing required columns. Available columns: {chunk.columns}"
                )
            logger.info(f"Loaded chunk {counter}")
            chunk[self.numeric_value] = pd.to_numeric(
                chunk[self.numeric_value], errors="coerce"
            )
            chunk = chunk.dropna(subset=[self.numeric_value])
            grouped = chunk.groupby(CODE)[self.numeric_value].apply(list).to_dict()

            for key, values in grouped.items():
                if key in lab_val_dict:
                    lab_val_dict[key].extend(values)
                else:
                    lab_val_dict[key] = values

            counter += 1
        return lab_val_dict

    def process_chunk(self, chunk):
        chunk[self.numeric_value] = chunk.apply(self.normalise, axis=1)
        return chunk

    def normalise(self, row):
        concept = row[CODE]
        value = row[self.numeric_value]
        if not pd.notnull(pd.to_numeric(value, errors="coerce")):
            return value
        else:
            value = pd.to_numeric(value)

        if self.normalisation_type == "Min_max":
            return self.min_max_normalise(concept, value)
        elif self.normalisation_type == "Quantiles":
            return self.quantile(concept, value)
        else:
            Warning(f"Normalisation type {self.normalisation_type} not implemented")

    def min_max_normalise(self, concept, value):
        if concept in self.min_max_vals:
            (min_val, max_val) = self.min_max_vals[concept]
            if max_val != min_val:
                normalised_value = (value - min_val) / (max_val - min_val)
                return round(max(0, min(1, normalised_value)), 3)
            else:
                return "UNIQUE"
        else:
            return "N/A"

    def quantile(self, concept, value):
        if concept not in self.quantiles:
            return "N/A"
        else:
            quantile_values = self.quantiles[concept]
            if len(quantile_values) != self.n_quantiles:
                raise ValueError(
                    f"Expected {self.n_quantiles} quantiles for concept '{concept}'"
                )
            for i, q in enumerate(quantile_values, start=1):
                if value <= q:
                    return f"Q{i}"
            return f"Q{self.n_quantiles}"
