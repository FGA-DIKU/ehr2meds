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

    def _try_read_csv_single(
        self,
        file_path: str,
        cols: Optional[List[str]] = None,
        sep: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> pd.DataFrame:
        """Helper to try reading CSV with given parameters, handling column selection."""
        try:
            kwargs = {}
            if sep:
                kwargs["sep"] = sep
            if encoding:
                kwargs["encoding"] = encoding

            if cols:
                try:
                    return pd.read_csv(file_path, usecols=cols, **kwargs)
                except (ValueError, KeyError):
                    # If usecols fails, read all columns first to check availability
                    df_all = pd.read_csv(file_path, **kwargs)
                    missing_cols = set(cols) - set(df_all.columns)
                    if missing_cols:
                        raise ValueError(
                            f"Columns not found in file: {missing_cols}. "
                            f"Available columns: {list(df_all.columns)}"
                        )
                    return df_all[cols]
            else:
                return pd.read_csv(file_path, **kwargs)
        except Exception:
            raise  # Re-raise to be caught by caller

    def _try_csv_read_strategies(
        self,
        file_path: str,
        cols: Optional[List[str]],
        read_func,
        context_msg: str = "",
        test_generator: bool = False,
    ):
        """Generic method to try CSV reading with 3-phase strategy: auto-detection -> separators -> encodings+separators."""
        last_error = None

        # Phase 1: Full auto-detection
        logger.debug(f"Trying pandas auto-detection{context_msg} for {file_path}")
        try:
            result = read_func(sep=None, encoding=None)
            if test_generator:
                # For generators, test by getting first item
                first_item = next(result)

                def gen():
                    yield first_item
                    yield from result

                logger.debug(f"Successfully read{context_msg} with auto-detection")
                return gen()
            logger.debug(f"Successfully read{context_msg} with auto-detection")
            return result
        except Exception as e:
            logger.debug(f"Auto-detection failed: {str(e)}")
            last_error = e

        # Phase 2: Try with explicit separators
        for separator in self.KNOWN_SEPARATORS:
            try:
                result = read_func(sep=separator, encoding=None)
                if test_generator:
                    first_item = next(result)

                    def gen():
                        yield first_item
                        yield from result

                    logger.debug(
                        f"Successfully read{context_msg} with separator '{separator}'"
                    )
                    return gen()
                logger.debug(
                    f"Successfully read{context_msg} with separator '{separator}'"
                )
                return result
            except Exception as e:
                logger.debug(f"Failed with separator '{separator}': {str(e)}")
                last_error = e

        # Phase 3: Try all encodings with all separators
        for encoding in self.CSV_ENCODINGS:
            for separator in self.KNOWN_SEPARATORS:
                try:
                    result = read_func(sep=separator, encoding=encoding)
                    if test_generator:
                        first_item = next(result)

                        def gen():
                            yield first_item
                            yield from result

                        logger.debug(
                            f"Successfully read{context_msg} with encoding '{encoding}' and separator '{separator}'"
                        )
                        return gen()
                    logger.debug(
                        f"Successfully read{context_msg} with encoding '{encoding}' and separator '{separator}'"
                    )
                    return result
                except Exception as e:
                    logger.debug(
                        f"Failed with encoding '{encoding}' and separator '{separator}': {str(e)}"
                    )
                    last_error = e

        error_msg = f"Unable to read file {repr(file_path)} with any method"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"
        raise ValueError(error_msg)

    def _load_csv(
        self, file_path: str, cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Try pandas auto-detection first, then fallback to explicit options."""
        if (
            file_path == "file_path"
            or not isinstance(file_path, str)
            or len(file_path) == 0
        ):
            raise ValueError(f"Invalid file_path provided: {repr(file_path)}")
        if not os.path.exists(file_path):
            raise ValueError(f"File does not exist: {file_path}")

        def read_func(sep=None, encoding=None):
            return self._try_read_csv_single(
                file_path, cols, sep=sep, encoding=encoding
            )

        return self._try_csv_read_strategies(file_path, cols, read_func)

    def _try_read_csv_chunks(
        self,
        file_path: str,
        cols: Optional[List[str]] = None,
        sep: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> Iterator[pd.DataFrame]:
        """Helper to try reading CSV chunks with given parameters."""
        kwargs = {"chunksize": self.chunksize}
        if sep:
            kwargs["sep"] = sep
        if encoding:
            kwargs["encoding"] = encoding
        if cols:
            kwargs["usecols"] = cols

        try:
            chunk_iter = pd.read_csv(file_path, **kwargs)
            for i, chunk in enumerate(chunk_iter):
                if self.test and i >= N_TEST_CHUNKS:
                    break
                yield chunk
        except (ValueError, KeyError) as col_error:
            if cols:
                # If usecols fails, try reading all columns then selecting
                kwargs.pop("usecols", None)
                chunk_iter = pd.read_csv(file_path, **kwargs)
                first_chunk = next(iter(chunk_iter))
                missing_cols = set(cols) - set(first_chunk.columns)
                if missing_cols:
                    raise ValueError(
                        f"Columns not found: {missing_cols}. "
                        f"Available: {list(first_chunk.columns)}"
                    )
                # Re-read and select columns
                chunk_iter = pd.read_csv(file_path, **kwargs)
                for i, chunk in enumerate(chunk_iter):
                    if self.test and i >= N_TEST_CHUNKS:
                        break
                    yield chunk[cols]
            else:
                raise
        except Exception:
            raise  # Re-raise to be caught by caller

    def _load_csv_chunks(
        self, file_path: str, cols: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        """Load CSV in chunks, trying pandas auto-detection first."""

        def read_func(sep=None, encoding=None):
            return self._try_read_csv_chunks(
                file_path, cols, sep=sep, encoding=encoding
            )

        chunk_iter = self._try_csv_read_strategies(
            file_path, cols, read_func, " chunks", test_generator=True
        )
        yield from chunk_iter

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
