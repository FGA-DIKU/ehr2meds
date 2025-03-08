import os
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

from ehr2meds.PREMEDS.preprocessing.constants import FILENAME
from ehr2meds.PREMEDS.preprocessing.io.azure import get_data_loader


@dataclass
class DataConfig:
    """Configuration for data handling"""

    output_dir: str
    file_type: str
    datastore: Optional[str] = None
    dump_path: Optional[str] = None
    chunksize: Optional[int] = None


class DataHandler:
    """Handles data loading and saving operations"""

    def __init__(self, config: DataConfig, logger, env: str, test: bool):
        self.output_dir = config.output_dir
        self.file_type = config.file_type
        self.env = env
        self.logger = logger
        self.test = test

        # Initialize the appropriate data loader
        self.data_loader = get_data_loader(
            env=self.env,
            datastore_name=config.datastore,
            dump_path=config.dump_path,
            chunksize=config.chunksize,
            logger=logger,
        )  # return azure or standard data loader

    def load_pandas(
        self, filename: str, cols: Optional[list[str]] = None
    ) -> pd.DataFrame:
        return self.data_loader.load_dataframe(
            filename=filename, test=self.test, test_rows=1_000_000, cols=cols
        )

    def load_chunks(self, cfg: dict) -> Iterator[pd.DataFrame]:
        cols = list(cfg.get("rename_columns", {}).keys())
        return self.data_loader.load_chunks(
            filename=cfg[FILENAME],
            cols=cols if len(cols) > 0 else None,
            test=self.test,
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
