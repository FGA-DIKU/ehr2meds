import logging
import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

N_TEST_CHUNKS = 2


class BaseDataLoader(ABC):
    PRIMARY_SEPARATOR = ";"
    FALLBACK_SEPARATOR = ","  # Only use coma as fallback, in the opposite case it will load the file with ";" and collapse all into one column
    CSV_ENCODINGS = ["iso88591", "utf8", "latin1"]

    def __init__(
        self,
        path: str,
        chunksize: Optional[int] = None,
        test: bool = False,
    ):
        self.path = path
        self.chunksize = chunksize
        self.test = test

    def load_dataframe(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Default implementation for loading entire files using pandas."""
        file_path = self._get_file_path(filename)
        self._check_file_exists(file_path)
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path, columns=cols)
        elif file_path.endswith((".csv", ".asc")):
            return self._load_csv(file_path, cols)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _detect_separator(self, file_path: str, encoding: str) -> str:
        """Detect the correct separator by checking first few lines."""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                first_line = f.readline().strip()
                # Count occurrences of each separator
                counts = {sep: first_line.count(sep) for sep in self.KNOWN_SEPARATORS}
                # Choose separator with most occurrences
                best_sep = max(counts.items(), key=lambda x: x[1])[0]
                if counts[best_sep] > 0:
                    return best_sep
        except Exception:
            pass
        return self.KNOWN_SEPARATORS[0]  # Default to first separator if detection fails

    def _load_csv(
        self, file_path: str, cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Try primary separator first, then fallback."""
        for encoding in self.CSV_ENCODINGS:
            # Try primary separator first
            try:
                return pd.read_csv(
                    file_path,
                    sep=self.PRIMARY_SEPARATOR,
                    encoding=encoding,
                    usecols=cols,
                )
            except Exception:
                # Try fallback separator
                try:
                    return pd.read_csv(
                        file_path,
                        sep=self.FALLBACK_SEPARATOR,
                        encoding=encoding,
                        usecols=cols,
                    )
                except Exception:
                    continue

        raise ValueError(f"Unable to read file {file_path} with any encoding")

    @abstractmethod
    def load_chunks(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        """Abstract method for chunk loading - implementations will differ."""
        pass

    def _get_file_path(self, filename: str) -> str:
        return join(self.path, filename)

    @staticmethod
    def _check_file_exists(file_path: str):
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")


class StandardDataLoader(BaseDataLoader):
    """Local file system data loader using pandas"""

    # Now only needs to implement chunk loading
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
        max_chunks = N_TEST_CHUNKS if self.test else float("inf")
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
            for sep in self.FALLBACK_SEPARATORS:
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
                            self.test and i >= N_TEST_CHUNKS
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
    """Azure-specific data loader using mltable for chunking"""

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
            for delimiter in self.FALLBACK_SEPARATORS:
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

    def load_chunks(
        self, filename: str, cols: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        import mltable

        tbl: mltable.MLTable = self._get_azure_dataset(filename)
        if cols:
            tbl = tbl.keep_columns(cols)

        max_chunks = N_TEST_CHUNKS if self.test else float("inf")
        chunks_processed = 0
        offset = 0

        while chunks_processed < max_chunks:
            logger.info(f"Loading chunk {chunks_processed}")
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
