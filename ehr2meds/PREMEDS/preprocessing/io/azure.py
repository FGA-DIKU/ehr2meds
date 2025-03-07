from os.path import join
from typing import Iterator, Optional

import pandas as pd


class StandardDataLoader:
    """Local file system data loader"""

    def __init__(self, dump_path: str, logger):
        self.dump_path = dump_path
        self.logger = logger

    def load_dataframe(
        self, filename: str, test: bool = False, n_rows: int = 1_000_000
    ) -> pd.DataFrame:
        file_path = join(self.dump_path, filename)

        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith((".csv", ".asc")):
            # Try different encodings
            for encoding in ["iso88591", "utf8", "latin1"]:
                try:
                    df = pd.read_csv(file_path, sep=";", encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Unable to read file {file_path} with any encoding")
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        if test:
            df = df.head(n_rows)
        return df

    def load_chunks(
        self, filename: str, chunk_size: int = 500_000, test: bool = False
    ) -> Iterator[pd.DataFrame]:
        file_path = join(self.dump_path, filename)

        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
            for i in range(0, len(df), chunk_size):
                if test and i >= 2 * chunk_size:
                    break
                yield df.iloc[i : i + chunk_size]
        elif file_path.endswith((".csv", ".asc")):
            for encoding in ["iso88591", "utf8", "latin1"]:
                try:
                    for chunk in pd.read_csv(
                        file_path, sep=";", encoding=encoding, chunksize=chunk_size
                    ):
                        if test:
                            yield chunk.head(chunk_size)
                            break
                        yield chunk
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Unable to read file {file_path} with any encoding")


class AzureDataLoader:
    """Azure-specific data loader"""

    def __init__(self, datastore: str, dump_path: str, logger):
        self.datastore = datastore
        self.dump_path = dump_path
        self.logger = logger

    def _get_azure_dataset(self, filename: str):
        from azureml.core import Dataset

        file_path = join(self.dump_path, filename)
        if filename.endswith(".parquet"):
            ds = Dataset.Tabular.from_parquet_files(path=(self.datastore, file_path))
        elif filename.endswith(".csv") or filename.endswith(".asc"):
            # Try different encodings
            encodings = ["iso88591", "utf8", "latin1"]
            for encoding in encodings:
                try:
                    ds = Dataset.Tabular.from_delimited_files(
                        path=(self.datastore, file_path),
                        separator=";",
                        encoding=encoding,
                    )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(
                    f"Unable to read file {file_path} with any of the provided encodings"
                )
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        return ds

    def load_dataframe(
        self,
        filename: str,
        test: bool = False,
        n_rows: int = 1_000_000,
        cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        ds = self._get_azure_dataset(filename)
        if test:
            ds = ds.take(n_rows)
        if cols:
            ds = ds.keep_columns(cols)
        return ds.to_pandas_dataframe()

    def load_chunks(
        self, filename: str, chunk_size: int = 500_000, test: bool = False
    ) -> Iterator[pd.DataFrame]:
        ds = self._get_azure_dataset(filename)
        i = 0
        max_chunks = 2 if test else float("inf")
        chunks_processed = 0

        while chunks_processed < max_chunks:
            self.logger.info(f"Loading chunk {i}")
            chunk = ds.skip(i * chunk_size).take(chunk_size)
            df = chunk.to_pandas_dataframe()
            if df.empty:
                break
            i += 1
            chunks_processed += 1
            yield df


def get_data_loader(
    env: str, datastore: Optional[str], dump_path: Optional[str], logger
):
    """Factory function to create the appropriate data loader"""
    if env == "azure":
        if datastore is None:
            raise ValueError("datastore must be provided when env is 'azure'")
        return AzureDataLoader(datastore, dump_path, logger)
    else:
        if dump_path is None:
            raise ValueError("dump_path must be provided when env is not 'azure'")
        return StandardDataLoader(dump_path, logger)
