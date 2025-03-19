import os
from typing import Iterator, Optional

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import FILENAME
from ehr2meds.PREMEDS.preprocessing.io.azure import get_data_loader


class DataHandler:
    """Handles data loading and saving operations.

    Args:
        output_dir: Directory path for output files
        file_type: Type of files to handle (e.g. 'csv', 'parquet')
        env: Environment type ('azure' or local)
        logger: Logger instance for logging messages
        datastore: Optional Azure datastore name for cloud storage
        dump_path: Optional local filesystem path for data files
        chunksize: Optional size of chunks for processing large files
        test: Optional flag to enable test mode with limited data
        test_rows: Optional number of rows to load in test mode
    """

    def __init__(
        self,
        output_dir: str,
        file_type: str,
        env: str,
        logger,
        datastore: Optional[str] = None,
        dump_path: Optional[str] = None,
        chunksize: Optional[int] = None,
        test: Optional[bool] = False,
        test_rows: Optional[int] = 1_000_000,
    ):
        self.output_dir = output_dir
        self.file_type = file_type
        self.env = env
        self.logger = logger

        # Initialize the appropriate data loader
        self.data_loader = get_data_loader(
            env=self.env,
            datastore_name=datastore,
            dump_path=dump_path,
            chunksize=chunksize,
            test=test,
            test_rows=test_rows,
            logger=logger,
        )  # return azure or standard data loader

    def load_pandas(
        self, filename: str, cols: Optional[list[str]] = None
    ) -> pd.DataFrame:
        return self.data_loader.load_dataframe(filename=filename, cols=cols)

    def load_chunks(self, cfg: dict) -> Iterator[pd.DataFrame]:
        cols = list(cfg.get("rename_columns", {}).keys())
        return self.data_loader.load_chunks(
            filename=cfg[FILENAME],
            cols=cols if len(cols) > 0 else None,
        )

    def save(self, df: pd.DataFrame, filename: str, mode: str = "w") -> None:
        """
        Save the processed data to a file.

        Args:
            df: DataFrame containing the processed data
            filename: Name of the file to save
            mode: Mode for saving the file ("w" for write, "a" for append)
        """
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {filename}, skipping save")
            return

        self.logger.info(f"Saving {filename} with {len(df)} rows")
        out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Decide on filetype
        file_type = self.file_type
        path = os.path.join(out_dir, f"{filename}.{file_type}")

        if file_type == "parquet":
            if mode == "w" or not os.path.exists(path):
                df.to_parquet(path, index=False)
            else:
                # For append mode with parquet, we need to read, concat, then write
                existing_df = pd.read_parquet(path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_parquet(path, index=False)
        elif file_type == "csv":
            if mode == "w":
                df.to_csv(path, index=False, mode="w")
            else:
                # append without header
                df.to_csv(path, index=False, mode="a", header=False)
        else:
            raise ValueError(f"Filetype {file_type} not implemented.")
