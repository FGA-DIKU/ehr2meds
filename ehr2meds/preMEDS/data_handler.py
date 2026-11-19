import logging
import os
import polars as pl
import pyarrow.parquet as pq
from typing import Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

# Common spellings of a missing value in CSV exports; these read as null.
CSV_NULL_VALUES = "#N/A|#N/A N/A|#NA|-1.#IND|-1.#QNAN|-NaN|-nan|1.#IND|1.#QNAN|<NA>|N/A|NA|NULL|NaN|None|n/a|nan|null".split(
    "|"
)


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

    def load(self, filename: str, cols: Mapping[str, Optional[str]]) -> pl.DataFrame:
        return next(self.read(filename, cols, chunksize=None))

    def load_chunks(self, filename: str, cols: Mapping[str, Optional[str]]) -> Iterator[pl.DataFrame]:
        return self.read(filename, cols, self.chunksize)

    @staticmethod
    def read(filename: str, cols: Mapping[str, Optional[str]], chunksize: Optional[int]) -> Iterator[pl.DataFrame]:
        """Yield the file in chunks of ``chunksize`` rows, or whole when it is None."""
        if filename.endswith(".parquet"):
            # pyarrow, not pl.scan_parquet: polars (1.41-1.42) reads time32 ms as seconds, nulling times after 00:01:26.
            if chunksize is None:
                chunks = [pl.from_arrow(pq.read_table(filename, columns=list(cols)))]
            else:
                batches = pq.ParquetFile(filename).iter_batches(columns=list(cols), batch_size=chunksize)
                chunks = (pl.from_arrow(batch) for batch in batches)
        elif filename.endswith((".csv", ".asc")):
            # Every CSV column is text, so IDs keep leading zeros and each chunk gets the same schema.
            lf = pl.scan_csv(filename, infer_schema=False, null_values=CSV_NULL_VALUES).select(list(cols))
            chunks = [lf.collect()] if chunksize is None else lf.collect_batches(chunk_size=chunksize)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        rename = {k: v for k, v in cols.items() if v is not None}
        # NaN counts as missing.
        return (chunk.rename(rename).with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None)) for chunk in chunks)

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

            self.writers[filename].write_table(table)

        elif self.write_file_type == "csv":
            first = not os.path.exists(path)
            with open(path, "w" if first else "a") as f:
                # Always write the time, even when a chunk is all midnight, so every chunk has the same format.
                df.write_csv(f, include_header=first, datetime_format="%Y-%m-%d %H:%M:%S%.f")

        else:
            raise ValueError(f"Filetype {self.write_file_type} not implemented.")

    def close(self):
        """
        Close all open Parquet writers.
        """

        for writer in self.writers.values():
            writer.close()

        self.writers.clear()
