import logging
import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Iterator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

N_TEST_CHUNKS = 2


class BaseDataLoader(ABC):
    KNOWN_SEPARATORS = [";", ","]
    CSV_ENCODINGS = ["utf-8-sig", "utf-8", "utf8", "iso88591", "latin1"]

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

    def _detect_separator(self, file_path: str) -> str:
        """Detect the correct separator by checking first few lines."""
        # Try all encodings to read the first line
        first_line = None
        for encoding in self.CSV_ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    first_line = f.readline()
                    break  # Successfully read, exit loop
            except (UnicodeDecodeError, Exception):
                continue  # Try next encoding
        
        # If we couldn't read with any encoding, default to comma
        if first_line is None:
            return ","
        
        # Count occurrences of each separator
        counts = {sep: first_line.count(sep) for sep in self.KNOWN_SEPARATORS}
        # Choose separator with most occurrences
        best_sep = max(counts.items(), key=lambda x: x[1])[0]
        if counts[best_sep] > 0:
            return best_sep
        # Default to comma if no separator found
        return ","

    def _load_csv(
        self, file_path: str, cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Try primary separator first, then fallback."""
        # Verify file_path is actually a path, not a variable name
        if file_path == "file_path" or not isinstance(file_path, str) or len(file_path) == 0:
            raise ValueError(f"Invalid file_path provided: {repr(file_path)}")
        
        # Verify file exists and is readable
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")
        
        last_error = None
        # Try UTF-8 encodings first to avoid misinterpreting UTF-8 files as iso88591
        utf8_encodings = ["utf-8-sig", "utf-8", "utf8"]
        other_encodings = [enc for enc in self.CSV_ENCODINGS if enc not in utf8_encodings]
        encoding_order = utf8_encodings + other_encodings
        
        # First, try UTF-8 encodings with all separators - if any succeeds, use it immediately
        for encoding in utf8_encodings:
            for separator in self.KNOWN_SEPARATORS:
                try:
                    if cols:
                        try:
                            df = pd.read_csv(
                                file_path,
                                sep=separator,
                                encoding=encoding,
                                usecols=cols,
                            )
                        except (ValueError, KeyError) as col_error:
                            # If usecols fails, try reading all columns first to see what's available
                            df_all = pd.read_csv(
                                file_path,
                                sep=separator,
                                encoding=encoding,
                            )
                            # Check which columns are missing
                            missing_cols = set(cols) - set(df_all.columns)
                            if missing_cols:
                                continue  # Try next separator/encoding
                            # If all columns exist, select them
                            df = df_all[cols]
                    else:
                        df = pd.read_csv(
                            file_path,
                            sep=separator,
                            encoding=encoding,
                        )
                    logger.debug(f"Successfully read {file_path} with UTF-8 encoding '{encoding}' and separator '{separator}'")
                    return df
                except Exception as e:
                    logger.debug(
                        f"Failed to read {file_path} with UTF-8 encoding '{encoding}' and separator '{separator}': {str(e)}"
                    )
                    continue
        
        # If UTF-8 encodings all failed, try other encodings
        for encoding in other_encodings:
            for separator in self.KNOWN_SEPARATORS:
                try:
                    if cols:
                        try:
                            df = pd.read_csv(
                                file_path,
                                sep=separator,
                                encoding=encoding,
                                usecols=cols,
                            )
                        except (ValueError, KeyError) as col_error:
                            df_all = pd.read_csv(
                                file_path,
                                sep=separator,
                                encoding=encoding,
                            )
                            missing_cols = set(cols) - set(df_all.columns)
                            if missing_cols:
                                raise ValueError(
                                    f"Columns not found in file: {missing_cols}. "
                                    f"Available columns: {list(df_all.columns)}"
                                )
                            df = df_all[cols]
                    else:
                        df = pd.read_csv(
                            file_path,
                            sep=separator,
                            encoding=encoding,
                        )
                    logger.debug(f"Successfully read {file_path} with encoding '{encoding}' and separator '{separator}'")
                    return df
                except Exception as e:
                    last_error = e
                    logger.debug(
                        f"Failed to read {file_path} with encoding '{encoding}' and separator '{separator}': {str(e)}"
                    )
                    continue
        
        # If all explicit encodings failed, try pandas' default encoding detection
        # (same as pd.read_csv without specifying encoding)
        logger.debug(f"Trying pandas default encoding detection for {file_path}")
        for separator in self.KNOWN_SEPARATORS:
            try:
                if cols:
                    try:
                        df = pd.read_csv(
                            file_path,
                            sep=separator,
                            usecols=cols,
                        )
                    except (ValueError, KeyError) as col_error:
                        df_all = pd.read_csv(
                            file_path,
                            sep=separator,
                        )
                        missing_cols = set(cols) - set(df_all.columns)
                        if missing_cols:
                            continue
                        df = df_all[cols]
                else:
                    df = pd.read_csv(
                        file_path,
                        sep=separator,
                    )
                logger.debug(f"Successfully read {file_path} with pandas default encoding and separator '{separator}'")
                return df
            except Exception as e:
                logger.debug(
                    f"Failed to read {file_path} with pandas default encoding and separator '{separator}': {str(e)}"
                )
                last_error = e
                continue
        
        # If we get here, all combinations failed
        # Use repr to ensure the actual path is shown
        error_msg = "Unable to read file: " + repr(file_path) + " with any encoding/separator combination"
        if last_error:
            error_msg += ". Last error: " + str(last_error)
        raise ValueError(error_msg)

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
        """Load CSV in chunks, trying separators in priority order."""
        # Try detected separator first, then fall back to trying all separators
        detected_separator = self._detect_separator(file_path)
        separators_to_try = [detected_separator] + [
            sep for sep in self.KNOWN_SEPARATORS if sep != detected_separator
        ]
        
        # Try UTF-8 encodings first
        utf8_encodings = ["utf-8-sig", "utf-8", "utf8"]
        other_encodings = [enc for enc in self.CSV_ENCODINGS if enc not in utf8_encodings]
        
        # First try UTF-8 encodings
        for encoding in utf8_encodings:
            for separator in separators_to_try:
                try:
                    chunk_iter = pd.read_csv(
                        file_path,
                        sep=separator,
                        encoding=encoding,
                        chunksize=self.chunksize,
                        usecols=cols,
                    )
                    for i, chunk in enumerate(chunk_iter):
                        if self.test and i >= N_TEST_CHUNKS:
                            break
                        yield chunk
                    return  # Exit if reading was successful
                except Exception as e:
                    logger.debug(
                        f"Failed with separator '{separator}' and UTF-8 encoding {encoding}: {str(e)}"
                    )
                    continue
        
        # Then try other encodings
        for encoding in other_encodings:
            for separator in separators_to_try:
                try:
                    chunk_iter = pd.read_csv(
                        file_path,
                        sep=separator,
                        encoding=encoding,
                        chunksize=self.chunksize,
                        usecols=cols,
                    )
                    for i, chunk in enumerate(chunk_iter):
                        if self.test and i >= N_TEST_CHUNKS:
                            break
                        yield chunk
                    return  # Exit if reading was successful
                except Exception as e:
                    logger.debug(
                        f"Failed with separator '{separator}' and encoding {encoding}: {str(e)}"
                    )
                    continue
        
        # Finally, try pandas' default encoding detection
        logger.debug(f"Trying pandas default encoding detection for chunks in {file_path}")
        for separator in separators_to_try:
            try:
                chunk_iter = pd.read_csv(
                    file_path,
                    sep=separator,
                    chunksize=self.chunksize,
                    usecols=cols,
                )
                for i, chunk in enumerate(chunk_iter):
                    if self.test and i >= N_TEST_CHUNKS:
                        break
                    yield chunk
                return  # Exit if reading was successful
            except Exception as e:
                logger.debug(
                    f"Failed with separator '{separator}' and pandas default encoding: {str(e)}"
                )
                continue

        raise ValueError(
            f"Unable to read file {file_path} with any encoding/separator combination"
        )


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
        delimiter = self._detect_separator(file_path)
        for encoding in self.CSV_ENCODINGS:
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
