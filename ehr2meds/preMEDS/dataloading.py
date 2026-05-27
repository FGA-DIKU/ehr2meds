import logging
import os
import pandas as pd
from abc import ABC
from os.path import join
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

N_TEST_CHUNKS = 2


class DataLoader(ABC):
    def __init__(
        self,
        path: str,
        chunksize: Optional[int] = None,
        test: bool = False,
    ):
        self.path = path
        self.chunksize = chunksize
        self.test = test

    def load_dataframe(self, filename: str, cols: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Default implementation for loading entire files using pandas."""
        file_path = self._get_file_path(filename)
        self._check_file_exists(file_path)
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path, columns=cols)
        elif file_path.endswith((".csv", ".asc")):
            return self._load_csv(file_path, cols, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _load_csv(self, file_path: str, cols: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        return pd.read_csv(file_path, usecols=cols, **kwargs)

    def load_chunks(self, filename: str, cols: Optional[List[str]] = None, **kwargs) -> Iterator[pd.DataFrame]:
        file_path = self._get_file_path(filename)
        self._check_file_exists(file_path)
        if file_path.endswith(".parquet"):
            yield from self._load_parquet_chunks(file_path, cols)
        elif file_path.endswith((".csv", ".asc")):
            yield from self._load_csv_chunks(file_path, cols, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _get_file_path(self, filename: str) -> str:
        return join(self.path, filename)

    @staticmethod
    def _check_file_exists(file_path: str):
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")

    def _load_parquet_chunks(self, file_path: str, cols: Optional[list[str]]) -> Iterator[pd.DataFrame]:
        import pyarrow.parquet as pq

        # Open the ParquetFile to enable reading by row groups
        pf = pq.ParquetFile(file_path)

        # Calculate how many row groups to read
        max_chunks = N_TEST_CHUNKS if self.test else float("inf")
        chunks_read = 0

        # Read and yield row groups
        for i, batch in enumerate(pf.iter_batches(batch_size=self.chunksize, columns=cols)):
            if chunks_read >= max_chunks:
                break
            df = batch.to_pandas()
            chunks_read += 1
            yield df

    def _load_csv_chunks(self, file_path: str, cols: Optional[List[str]] = None, **kwargs) -> Iterator[pd.DataFrame]:
        """Load CSV in chunks, using pandas read_csv."""
        chunk_iter = pd.read_csv(file_path, chunksize=self.chunksize, usecols=cols, **kwargs)
        for i, chunk in enumerate(chunk_iter):
            if self.test and i >= N_TEST_CHUNKS:
                break
            yield chunk
        return  # Exit if reading was successful
