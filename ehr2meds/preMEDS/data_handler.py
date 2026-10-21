import logging
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


class DataHandler:
    """Handles data loading and saving operations.

    Args:
        output_dir: Directory path for output files
        write_file_type: Type of files to write (e.g. 'csv', 'parquet')
        chunksize: Optional size of chunks for processing large files
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        write_file_type: str = "parquet",
        chunksize: int = 1_000_000,
    ):
        self.output_dir = output_dir
        self.write_file_type = write_file_type
        self.chunksize = chunksize
        self._writers = {}

    def load(self, filename: str, cols: Mapping[str, Optional[str]]) -> pd.DataFrame:
        rename = {k: v for k, v in cols.items() if v is not None}
        if filename.endswith(".parquet"):
            return pd.read_parquet(filename, columns=list(cols)).rename(columns=rename)
        elif filename.endswith((".csv", ".asc")):
            return pd.read_csv(filename, usecols=list(cols), engine="python").rename(columns=rename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def load_chunks(self, filename: str, cols: Mapping[str, Optional[str]]):
        rename = {k: v for k, v in cols.items() if v is not None}
        if filename.endswith(".parquet"):
            pf = pq.ParquetFile(filename)
            for batch in pf.iter_batches(columns=list(cols), batch_size=self.chunksize):
                yield batch.to_pandas().rename(columns=rename)
        elif filename.endswith((".csv", ".asc")):
            for chunk in pd.read_csv(filename, usecols=list(cols), chunksize=self.chunksize, engine="python"):
                yield chunk.rename(columns=rename)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    def save(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data incrementally to a single Parquet file.
        """

        if df.empty:
            logger.warning(
                f"Empty DataFrame for {filename}, skipping save"
            )
            return

        if self.output_dir is None:
            raise AttributeError("output_dir is not set")

        logger.info(f"Saving {filename} with {len(df):_} rows")

        os.makedirs(self.output_dir, exist_ok=True)

        path = os.path.join(
            self.output_dir,
            f"{filename}.{self.write_file_type}"
        )

        if self.write_file_type == "parquet":

            table = pa.Table.from_pandas(
                df,
                preserve_index=False
            )

            if filename not in self._writers:
                self._writers[filename] = pq.ParquetWriter(
                    path,
                    table.schema,
                    compression="snappy"
                )

            self._writers[filename].write_table(table)

        elif self.write_file_type == "csv":

            if not os.path.exists(path):
                df.to_csv(path, index=False)
            else:
                df.to_csv(
                    path,
                    index=False,
                    mode="a",
                    header=False
                )

        else:
            raise ValueError(
                f"Filetype {self.write_file_type} not implemented."
            )
        
    def close(self):
        """
        Close all open Parquet writers.
        """

        for writer in self._writers.values():
            writer.close()

        self._writers.clear()
