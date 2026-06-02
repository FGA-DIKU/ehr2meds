import logging
import numpy as np
import pandas as pd
from ehr2meds.preMEDS.constants import CODE
from ehr2meds.preMEDS.dataloading import DataLoader
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class Normalizer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.test = cfg.test
        logger.info(f"test {self.test}")
        self.normalization_type = cfg.data["norm_type"]
        self.data_loader = DataLoader(
            chunksize=self.cfg.data.chunksize,
            test=self.test,
        )
        # Initialize distribution data placeholders
        self.min_max_vals = None
        self.numeric_value = cfg.data["numeric_value"]

    def __call__(self):
        print("Getting lab distribution")
        if self.cfg.data.get("dist_path", None):
            dist = pd.read_csv(self.cfg.data.dist_path)
        else:
            dist = self.get_lab_values(
                filename=self.cfg.file_name, 
                input_path=self.cfg.paths.input,
                numeric_value=self.cfg.data.numeric_value,
                build_distribution_on=self.cfg.data.build_distribution_on,
                norm_type=self.cfg.data.norm_type,
            )
        print("Distribution data loaded")
        self.process_distribution_data(dist)
        print("Normalizing data")
        self.normalize_data()

    def normalize_data(self):
        input_root = Path(self.cfg.paths.input)
        output_path = Path(self.cfg.paths.output)
        base_path = input_root / "data"
        for file_path in sorted(base_path.glob("*/*.parquet")):
            for chunk in tqdm(
                self.data_loader.load_chunks(filename=str(file_path)),
                desc=f"Processing {file_path}",
            ):
                chunk = self.normalize_chunk(chunk)
                rel_path = file_path.relative_to(base_path)
                self.save_chunk(chunk, rel_path, output_path)


    def process_distribution_data(self, dist: Dict[str, List[float]]) -> None:
        """Process distribution data based on normalization type."""
        if self.normalization_type == "min_max":
            self.process_minmax_distribution(dist)
        else:
            raise ValueError("Invalid type of normalization")

    def process_minmax_distribution(self, dist: Dict[str, List[float]]) -> None:
        """Process distribution data for min-max normalization."""
        self.min_max_vals = {
            concept: (
                (np.percentile(dist[concept], 0.01 * 100) if len(dist[concept]) > 1 else dist[concept][0]),
                (np.percentile(dist[concept], 0.99 * 100) if len(dist[concept]) > 1 else dist[concept][0]),
            )
            for concept in dist
            if dist[concept]
        }

    def save_chunk(
        self, chunk: pd.DataFrame, file_path: Path, output_path: Path
    ) -> None:
        """Save a processed chunk under output_path, preserving relative file_path."""
        save_path = output_path / file_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out = chunk.copy()
        if save_path.suffix == ".parquet" and out[self.numeric_value].dtype == object:
            out[self.numeric_value] = out[self.numeric_value].astype("string")
        if save_path.suffix == ".parquet":
            out.to_parquet(save_path, index=False)
        else:
            out.to_csv(save_path, index=False)

    def get_lab_values(self, input_path: str, filename: str,   norm_type: str, numeric_value: str, build_distribution_on: List[str]):
        logger.info("Getting lab distribution")
        lab_val_dict = {}
        counter = 0

        for split in build_distribution_on:
            base_path = Path(input_path) / "convert_to_sharded_events" / split
            table_name = Path(filename).stem
            for file_path in sorted(base_path.glob(f"*/{table_name}/*.parquet")):
                for chunk in tqdm(
                    self.data_loader.load_chunks(filename=str(file_path)),
                    desc="Building lab distribution",
                ):
                    if self.numeric_value not in chunk.columns or CODE not in chunk.columns:
                        raise ValueError(f"Missing required columns. Available columns: {chunk.columns}")
                    logger.info(f"Loaded {self.cfg.data.chunksize * counter}")
                    chunk[self.numeric_value] = pd.to_numeric(chunk[self.numeric_value], errors="coerce")
                    chunk = chunk.dropna(subset=[self.numeric_value])
                    grouped = chunk.groupby(CODE)[self.numeric_value].apply(list).to_dict()

                    for key, values in grouped.items():
                        if key in lab_val_dict:
                            lab_val_dict[key].extend(values)
                        else:
                            lab_val_dict[key] = values

                    counter += 1
        return lab_val_dict

    def normalize_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        nums = pd.to_numeric(chunk[self.numeric_value], errors="coerce")
        if self.normalization_type == "Min_max":
            return self.min_max_normalize(chunk, nums)
        raise ValueError(f"Invalid normalization type: {self.normalization_type}")

    def min_max_normalize(
        self, chunk: pd.DataFrame, nums: pd.Series
    ) -> pd.DataFrame:
        """Min-max scale numeric values for codes present in min_max_vals."""
        codes = chunk[CODE]
        for code, (min_val, max_val) in self.min_max_vals.items():
            mask = nums.notna() & (codes == code)
            if not mask.any() or max_val == min_val:
                continue
            scaled = (nums.loc[mask] - min_val) / (max_val - min_val)
            chunk.loc[mask, self.numeric_value] = np.clip(
                scaled, 0, 1
            ).to_numpy()
        return chunk