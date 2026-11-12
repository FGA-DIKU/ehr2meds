import logging
import os
import polars as pl
import pyarrow.parquet as pq
from typing import Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

# Matches pandas' CSV text, so CSV preMEDS output reads the same downstream.
CSV_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S%.f"


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
        self.writers = {}

    @staticmethod
    def scan(filename: str, cols: Mapping[str, Optional[str]]) -> pl.LazyFrame:
        if filename.endswith(".parquet"):
            lf = pl.scan_parquet(filename)
        elif filename.endswith((".csv", ".asc")):
            # Every CSV column is text, so IDs keep leading zeros and each chunk gets the same schema.
            lf = pl.scan_csv(filename, infer_schema=False)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        rename = {k: v for k, v in cols.items() if v is not None}
        # NaN counts as missing, as it did in pandas.
        return lf.select(list(cols)).rename(rename).with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))

    def load(self, filename: str, cols: Mapping[str, Optional[str]]) -> pl.DataFrame:
        return self.scan(filename, cols).collect()

    def load_chunks(self, filename: str, cols: Mapping[str, Optional[str]]) -> Iterator[pl.DataFrame]:
        return self.scan(filename, cols).collect_batches(chunk_size=self.chunksize)

    def save(self, df: pl.DataFrame, filename: str) -> None:
        """
        Save processed data incrementally to a single file.
        """

        if df.is_empty():
            logger.warning(f"Empty DataFrame for {filename}, skipping save")
            return

        if self.output_dir is None:
            raise AttributeError("output_dir is not set")

        logger.info(f"Saving {filename} with {df.height:_} rows")

        os.makedirs(self.output_dir, exist_ok=True)

        path = os.path.join(self.output_dir, f"{filename}.{self.write_file_type}")

        if self.write_file_type == "parquet":
            table = df.to_arrow()

            if filename not in self.writers:
                self.writers[filename] = pq.ParquetWriter(path, table.schema, compression="snappy")
            writer = self.writers[filename]

            # An all-null chunk column has null type; cast it to the file's type instead of failing.
            if table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)

        elif self.write_file_type == "csv":
            first = not os.path.exists(path)
            with open(path, "w" if first else "a") as f:
                df.write_csv(f, include_header=first, datetime_format=CSV_DATETIME_FORMAT)

        else:
            raise ValueError(f"Filetype {self.write_file_type} not implemented.")

    def close(self):
        """
        Close all open Parquet writers.
        """

        for writer in self.writers.values():
            writer.close()

        self.writers.clear()
