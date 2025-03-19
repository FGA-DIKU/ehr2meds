import logging
import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    CSV_ENCODINGS = ["iso88591", "utf8", "latin1"]
    CSV_SEPARATORS = [";", ","]

    def __init__(self, path: str, chunksize: Optional[int] = None, test: bool = False):
        self.path = path
        self.chunksize = chunksize
        self.test = test

    @abstractmethod
    def load_dataframe(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_chunks(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        pass

    def _get_file_path(self, filename: str) -> str:
        return join(self.path, filename)

    @staticmethod
    def _check_file_exists(file_path: str):
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")


class StandardDataLoader(BaseDataLoader):
    """Local file system data loader using pandas"""

    def load_dataframe(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        file_path = self._get_file_path(filename)
        self._check_file_exists(file_path)
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path, columns=cols)
        elif file_path.endswith((".csv", ".asc")):
            df = self._load_csv(file_path, cols)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        return df

    @classmethod
    def _load_csv(
        cls, file_path: str, cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        df = None
        for encoding in cls.CSV_ENCODINGS:
            for sep in cls.CSV_SEPARATORS:
                try:
                    df = pd.read_csv(
                        file_path, sep=sep, encoding=encoding, usecols=cols
                    )
                    break  # Successful read; break out of inner loop.
                except Exception:
                    continue
            if df is not None:
                break  # Exit outer loop if a valid DataFrame was read.
        if df is None:
            raise ValueError(f"Unable to read file {file_path} with any encoding")
        return df

    def load_chunks(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        file_path = self._get_file_path(filename)
        self._check_file_exists(file_path)
        if file_path.endswith(".parquet"):
            yield from self._load_parquet_chunks(file_path, cols)
        elif file_path.endswith((".csv", ".asc")):
            yield from self._load_csv_chunks(file_path, cols)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _load_parquet_chunks(
        self, file_path: str, cols: Optional[list[str]]
    ) -> Iterator[pd.DataFrame]:
        import pyarrow.parquet as pq

        # Open the ParquetFile to enable reading by row groups
        pf = pq.ParquetFile(file_path)

        # Calculate how many row groups to read
        max_chunks = 3 if self.test else float("inf")
        chunks_read = 0

        # Read and yield row groups
        for i, batch in enumerate(
            pf.iter_batches(batch_size=self.chunksize, columns=cols)
        ):
            if chunks_read >= max_chunks:
                break
            df = batch.to_pandas()
            chunks_read += 1
            yield df

    def _load_csv_chunks(
        self, file_path: str, cols: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        for encoding in self.CSV_ENCODINGS:
            for sep in self.CSV_SEPARATORS:
                try:
                    chunk_iter = pd.read_csv(
                        file_path,
                        sep=sep,
                        encoding=encoding,
                        chunksize=self.chunksize,
                        usecols=cols,
                    )
                    for i, chunk in enumerate(chunk_iter):
                        if (
                            self.test and i >= 3
                        ):  # In test mode, yield only first three chunks.
                            break
                        yield chunk
                    return  # Exit if reading was successful.
                except Exception as e:
                    logger.info(
                        f"Failed with encoding {encoding} and sep {sep}: {str(e)}"
                    )
                    continue
        raise ValueError(f"Unable to read file {file_path} with any encoding")


class AzureDataLoader(BaseDataLoader):
    """Azure-specific data loader using mltable"""

    def __init__(
        self,
        path: str,
        chunksize: Optional[int] = None,
        test: bool = False,
        test_rows: int = 1_000_000,
    ):
        super().__init__(path, chunksize, test)
        self.test_rows = test_rows

    def _get_azure_dataset(self, filename: str):
        import mltable

        logger.info(f"Loading dataset from {self.path}")
        file_path = self._get_file_path(filename)
        self._check_file_exists(file_path)
        if filename.endswith(".parquet"):
            return mltable.from_parquet_files([{"file": file_path}])
        elif filename.endswith((".csv", ".asc")):
            return self._get_csv_dataset(filename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def _get_csv_dataset(self, filename: str):
        import mltable

        file_path = self._get_file_path(filename)
        self._check_file_exists(file_path)
        for encoding in self.CSV_ENCODINGS:
            for delimiter in self.CSV_SEPARATORS:
                try:
                    return mltable.from_delimited_files(
                        [{"file": file_path}],
                        delimiter=delimiter,
                        encoding=encoding,
                    )
                except Exception as e:
                    logger.info(
                        f"Failed with encoding {encoding} and sep {delimiter}: {str(e)}"
                    )
                    continue
        raise ValueError(
            f"Unable to read file {filename} with any of the provided encodings and delimiters"
        )

    def load_dataframe(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        import mltable

        tbl: mltable.MLTable = self._get_azure_dataset(filename)
        if self.test:
            tbl = tbl.take(self.test_rows)
        if cols:
            tbl = tbl.keep_columns(cols)
        return tbl.to_pandas_dataframe()

    def load_chunks(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        import mltable

        tbl: mltable.MLTable = self._get_azure_dataset(filename)
        if cols:
            tbl = tbl.keep_columns(cols)

        max_chunks = 3 if self.test else float("inf")
        chunks_processed = 0
        offset = 0

        while chunks_processed < max_chunks:
            logger.info(f"Loading chunk {chunks_processed}")
            # Take directly from the current offset instead of using skip with i * chunksize
            chunk = (
                tbl.skip(offset).take(self.chunksize)
                if offset > 0
                else tbl.take(self.chunksize)
            )
            df = chunk.to_pandas_dataframe()
            if df.empty:
                break

            offset += self.chunksize
            chunks_processed += 1
            yield df


def get_data_loader(
    env: str, path: str, chunksize: Optional[int], test: bool, test_rows: Optional[int]
):
    """Factory function to create the appropriate data loader"""
    if env == "azure":
        return AzureDataLoader(path, chunksize, test, test_rows)
    else:
        return StandardDataLoader(path, chunksize, test)
