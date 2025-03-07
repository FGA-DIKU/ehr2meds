from os.path import join
from typing import Iterator, Optional

import pandas as pd


class StandardDataLoader:
    """Local file system data loader"""

    def __init__(self, dump_path: str, logger, chunksize: int = None):
        self.dump_path = dump_path
        self.logger = logger
        self.chunksize = chunksize

    def load_dataframe(
        self, filename: str, test: bool = False, test_rows: int = 1_000_000, cols=None
    ) -> pd.DataFrame:
        file_path = join(self.dump_path, filename)
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path, columns=cols)
        elif file_path.endswith((".csv", ".asc")):
            df = StandardDataLoader._load_csv(file_path, cols)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        if test:
            if df is not None:
                df = df.head(test_rows)
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
        self, filename: str, cols: Optional[list[str]] = None, test: bool = False
    ) -> Iterator[pd.DataFrame]:
        file_path = join(self.dump_path, filename)

        if file_path.endswith(".parquet"):
            yield from self._load_parquet_chunks(file_path, cols, test)
        elif file_path.endswith((".csv", ".asc")):
            yield from self._load_csv_chunks(file_path, cols, test)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _load_parquet_chunks(
        self, file_path: str, cols: Optional[list[str]], test: bool
    ) -> Iterator[pd.DataFrame]:
        df = pd.read_parquet(file_path, columns=cols)
        if test:
            # In test mode, only yield up to two chunks based on self.chunksize
            for i in range(0, len(df), self.chunksize):
                if i >= 3 * self.chunksize:
                    break
                yield df.iloc[i : i + self.chunksize]
        else:
            yield df

    def _load_csv_chunks(
        self, file_path: str, cols: Optional[list[str]], test: bool
    ) -> Iterator[pd.DataFrame]:
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
                            test and i >= 3
                        ):  # In test mode, yield only the first three chunks
                            break
                        yield chunk
                    # If we reached here without an exception, reading succeeded; exit the function.
                    return
                except Exception as e:
                    self.logger.info(
                        f"Failed with encoding {encoding} and sep {sep}: {str(e)}"
                    )
                    continue
        raise ValueError(f"Unable to read file {file_path} with any encoding")


class AzureDataLoader:
    """Azure-specific data loader"""

    def __init__(
        self, datastore_name: str, dump_path: str, logger, chunksize: int = None
    ):
        from ehr2meds.PREMEDS.azure_run import datastore

        self.datastore = datastore(name=datastore_name)
        self.dump_path = dump_path
        self.logger = logger
        self.chunksize = chunksize

    def _get_azure_dataset(self, filename: str):
        from azureml.core import Dataset

        file_path = join(self.dump_path, filename)
        self.logger.info(f"Loading dataset from {self.datastore} at {file_path}")
        if filename.endswith(".parquet"):
            return Dataset.Tabular.from_parquet_files(path=(self.datastore, file_path))
        elif filename.endswith((".csv", ".asc")):
            return self._get_csv_dataset(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _get_csv_dataset(self, file_path: str):
        from azureml.core import Dataset

        encodings = ["iso88591", "utf8", "latin1"]
        delimiters = [";", ","]
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    return Dataset.Tabular.from_delimited_files(
                        path=(self.datastore, file_path),
                        separator=delimiter,
                        encoding=encoding,
                    )
                except Exception as e:
                    self.logger.info(
                        f"Failed with encoding {encoding} and sep {delimiter}: {str(e)}"
                    )
                    continue
        raise ValueError(
            f"Unable to read file {file_path} with any of the provided encodings and delimiters"
        )

    def load_dataframe(
        self,
        filename: str,
        test: bool = False,
        test_rows: int = 1_000_000,
        cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        ds = self._get_azure_dataset(filename)
        if test:
            ds = ds.take(test_rows)
        if cols:
            ds = ds.keep_columns(cols)
        return ds.to_pandas_dataframe()

    def load_chunks(self, filename: str, test: bool = False) -> Iterator[pd.DataFrame]:
        ds = self._get_azure_dataset(filename)
        i = 0
        max_chunks = 2 if test else float("inf")
        chunks_processed = 0

        while chunks_processed < max_chunks:
            self.logger.info(f"Loading chunk {i}")
            chunk = ds.skip(i * self.chunksize).take(self.chunksize)
            df = chunk.to_pandas_dataframe()
            if df.empty:
                break
            i += 1
            chunks_processed += 1
            yield df


def get_data_loader(
    env: str,
    datastore_name: Optional[str],
    dump_path: Optional[str],
    chunksize: Optional[int],
    logger,
):
    """Factory function to create the appropriate data loader"""
    if env == "azure":
        if datastore_name is None:
            raise ValueError("datastore_name must be provided when env is 'azure'")
        return AzureDataLoader(datastore_name, dump_path, logger, chunksize)
    else:
        if dump_path is None:
            raise ValueError("dump_path must be provided when env is not 'azure'")
        return StandardDataLoader(dump_path, logger, chunksize)
