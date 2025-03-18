import os
from os.path import join
from typing import Iterator, Optional

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StandardDataLoader:
    """Local file system data loader"""

    def __init__(self, path: str, chunksize: int = None, test: bool = False):
        self.path = path
        self.chunksize = chunksize
        self.test = test

    def load_dataframe(self, filename: str, cols=None) -> pd.DataFrame:
        file_path = join(self.path, filename)
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path, columns=cols)
        elif file_path.endswith((".csv", ".asc")):
            df = self._load_csv(file_path, cols)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        return df

    @staticmethod
    def _load_csv(file_path: str, cols: Optional[list[str]] = None) -> pd.DataFrame:
        separators = [";", ","]
        encodings = ["iso88591", "utf8", "latin1"]
        df = None
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        file_path, sep=sep, encoding=encoding, usecols=cols
                    )
                    break  # If successful, break out of the inner loop
                except Exception:
                    continue
            if df is not None:
                break  # Break out of the outer loop if a valid DataFrame was read
        if df is None:
            raise ValueError(f"Unable to read file {file_path} with any encoding")
        return df

    def load_chunks(
        self, filename: str, cols: Optional[list[str]] = None
    ) -> Iterator[pd.DataFrame]:
        file_path = join(self.path, filename)

        if file_path.endswith(".parquet"):
            yield from self._load_parquet_chunks(file_path, cols)
        elif file_path.endswith((".csv", ".asc")):
            yield from self._load_csv_chunks(file_path, cols)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _load_parquet_chunks(
        self, file_path: str, cols: Optional[list[str]]
    ) -> Iterator[pd.DataFrame]:
        df = pd.read_parquet(file_path, columns=cols)
        if self.test:
            # In test mode, only yield up to two chunks based on self.chunksize for consistency with csv
            for i in range(0, len(df), self.chunksize):
                if i >= 3 * self.chunksize:
                    break
                yield df.iloc[i : i + self.chunksize]
        else:
            yield df

    def _load_csv_chunks(
        self, file_path: str, cols: Optional[list[str]]
    ) -> Iterator[pd.DataFrame]:
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        for encoding in ["iso88591", "utf8", "latin1"]:
            for sep in [";", ","]:
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
                        ):  # In test mode, yield only the first three chunks
                            break
                        yield chunk
                    # If we reached here without an exception, reading succeeded; exit the function.
                    return
                except Exception as e:
                    logger.info(
                        f"Failed with encoding {encoding} and sep {sep}: {str(e)}"
                    )
                    continue
        raise ValueError(f"Unable to read file {file_path} with any encoding")


class AzureDataLoader:
    """Azure-specific data loader"""

    def __init__(
        self,
        path: str,
        chunksize: int = None,
        test: bool = False,
        test_rows: int = 1_000_000,
    ):
        self.path = path
        self.chunksize = chunksize
        self.test = test
        self.test_rows = (
            test_rows  # used for testing load_dataframe (e.g. patient data)
        )

    def _get_azure_dataset(self, filename: str):
        import mltable

        logger.info(f"Loading dataset from {self.path}")
        if filename.endswith(".parquet"):
            return mltable.from_parquet_files([{"file": join(self.path, filename)}])
        elif filename.endswith((".csv", ".asc")):
            return self._get_csv_dataset(filename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def _get_csv_dataset(self, file_path: str):
        import mltable

        if not os.path.exists(join(self.path, file_path)):
            raise ValueError(f"File {file_path} does not exist")
        encodings = ["iso88591", "utf8", "latin1"]
        delimiters = [";", ","]
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    return mltable.from_delimited_files(
                        path=join(self.path, file_path),
                        separator=delimiter,
                        encoding=encoding,
                    )
                except Exception as e:
                    logger.info(
                        f"Failed with encoding {encoding} and sep {delimiter}: {str(e)}"
                    )
                    continue
        raise ValueError(
            f"Unable to read file {file_path} with any of the provided encodings and delimiters"
        )

    def load_dataframe(
        self,
        filename: str,
        cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        import mltable

        tbl: mltable.MLTable = self._get_azure_dataset(filename)
        if self.test:
            tbl = tbl.take(self.test_rows)
        if cols:
            tbl = tbl.keep_columns(cols)
        return tbl.to_pandas_dataframe()

    def load_chunks(
        self, filename: str, cols: Optional[list[str]] = None
    ) -> Iterator[pd.DataFrame]:
        import mltable

        tbl: mltable.MLTable = self._get_azure_dataset(filename)
        if cols:
            tbl = tbl.keep_columns(cols)
        i = 0
        max_chunks = 3 if self.test else float("inf")
        chunks_processed = 0

        while chunks_processed < max_chunks:
            logger.info(f"Loading chunk {i}")
            chunk = tbl.skip(i * self.chunksize).take(self.chunksize)
            df = chunk.to_pandas_dataframe()
            if df.empty:
                break
            i += 1
            chunks_processed += 1
            yield df


def get_data_loader(
    env: str,
    path: str,
    chunksize: Optional[int],
    test: bool,
    test_rows: Optional[int],
):
    """Factory function to create the appropriate data loader"""
    if env == "azure":
        return AzureDataLoader(path, chunksize, test, test_rows)
    else:
        return StandardDataLoader(path, chunksize, test)
