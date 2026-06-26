import logging
import os
import pandas as pd
import pyarrow.parquet as pq
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


class DataHandler:
    """Handles data loading and saving operations.

    Args:
        output_dir: Directory path for output files
        file_type: Type of files to handle (e.g. 'csv', 'parquet')
        chunksize: Optional size of chunks for processing large files
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        file_type: str = "parquet",
        chunksize: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.file_type = file_type
        if chunksize is None:
            chunksize = 500_000
        self.chunksize = chunksize

    def load(self, filename: str, cols: Mapping[str, Optional[str]]) -> pd.DataFrame:
        rename = {k: v for k, v in cols.items() if v is not None}
        if filename.endswith(".parquet"):
            return pd.read_parquet(filename, columns=list(cols)).rename(columns=rename)
        elif filename.endswith((".csv", ".asc")):
            return pd.read_csv(filename, usecols=list(cols)).rename(columns=rename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def load_chunks(self, filename: str, cols: Mapping[str, Optional[str]]):
        rename = {k: v for k, v in cols.items() if v is not None}
        if filename.endswith(".parquet"):
            pf = pq.ParquetFile(filename)
            for batch in pf.iter_batches(columns=list(cols), batch_size=self.chunksize):
                yield batch.to_pandas().rename(columns=rename)
        elif filename.endswith((".csv", ".asc")):
            for chunk in pd.read_csv(filename, usecols=list(cols), chunksize=self.chunksize):
                yield chunk.rename(columns=rename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def save(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save the processed data to a file.

        Args:
            df: DataFrame containing the processed data
            filename: Name of the file to save
            mode: Mode for saving the file ("w" for write, "a" for append)
        """
        if df.empty:
            logger.error(f"Empty DataFrame for {filename}, skipping save")
            return
        if self.output_dir is None:
            raise AttributeError("`output_dir` is not set; define `output_dir` for .save calls")

        logger.info(f"Saving {filename} with {len(df):_} rows")
        os.makedirs(self.output_dir, exist_ok=True)

        # Decide on filetype
        path = os.path.join(self.output_dir, f"{filename}.{self.file_type}")

        if self.file_type == "parquet":
            if not os.path.exists(path):
                df.to_parquet(path, index=False)
            else:
                # For append mode with parquet, we need to read, concat, then write
                existing_df = pd.read_parquet(path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_parquet(path, index=False)
        elif self.file_type == "csv":
            if not os.path.exists(path):
                df.to_csv(path, index=False, mode="w")
            else:
                # append without header
                df.to_csv(path, index=False, mode="a", header=False)
        else:
            raise ValueError(f"Filetype {self.file_type} not implemented.")
