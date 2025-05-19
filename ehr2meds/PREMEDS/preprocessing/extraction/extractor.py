import logging
import os
from os.path import split, dirname
from typing import Dict, Any

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import CODE
from ehr2meds.PREMEDS.preprocessing.io.dataloader import get_data_loader

logger = logging.getLogger(__name__)


class ValueExtractor:
    """
    Extracts for each lab 'code' a list of its numeric values,
    streaming through the input via pandas for CSVs and using
    the data loader for Parquet.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.test = cfg.test
        logger.info(f"test mode = {self.test}")
        self.input_path = cfg.paths.input
        self.chunksize = cfg.data.chunksize
        self.numeric_value = cfg.data.numeric_value
        # Prepare parquet loader; will be used if extension is .parquet
        path = dirname(self.input_path)
        filename = split(self.input_path)[1]
        self.parquet_loader = get_data_loader(
            env=cfg.env,
            path=path,
            chunksize=self.chunksize,
            test=self.test,
            test_rows=cfg.data.get("test_rows", 100_000),
        )

    def __call__(self) -> Dict[str, Any]:
        logger.info("Getting lab distribution")
        return self.get_lab_values()

    def get_lab_values(self) -> Dict[str, list]:
        filepath = self.input_path
        ext = os.path.splitext(filepath)[1].lower()
        lab_val_dict: Dict[str, list] = {}

        # Choose iterator based on file extension
        if ext == ".parquet":
            # Use existing data loader for parquet
            filename = split(filepath)[1]
            chunk_iter = self.parquet_loader.load_chunks(
                filename=filename,
                cols=[self.numeric_value, CODE],
            )
        elif ext in (".csv", ".asc"):  # infer CSV from path
            # Use pandas for CSV streaming
            chunk_iter = pd.read_csv(
                filepath,
                usecols=[self.numeric_value, CODE],
                sep=None,
                engine="python",
                chunksize=self.chunksize,
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # optional chunk limit in test mode
        max_chunks = (
            self.cfg.data.get("test_chunks", float("inf"))
            if self.test
            else float("inf")
        )

        for i, chunk in enumerate(chunk_iter):
            if i >= max_chunks:
                break

            # ensure columns
            if self.numeric_value not in chunk.columns or CODE not in chunk.columns:
                raise ValueError(
                    f"Missing required columns. Got: {chunk.columns.tolist()}"
                )

            logger.info(f"Processing chunk {i}, raw rows = {len(chunk)}")

            # coerce and drop
            chunk[self.numeric_value] = pd.to_numeric(
                chunk[self.numeric_value], errors="coerce"
            )
            chunk = chunk.dropna(subset=[self.numeric_value])
            logger.info(f"  after dropna = {len(chunk)} rows")

            # group by code
            grouped = chunk.groupby(CODE)[self.numeric_value].apply(list).to_dict()

            # merge
            for code, vals in grouped.items():
                lab_val_dict.setdefault(code, []).extend(vals)

        return lab_val_dict
