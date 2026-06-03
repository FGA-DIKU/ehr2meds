import logging
import numpy as np
import pandas as pd
from ehr2meds.preMEDS.constants import CODE, SUBJECT_ID, TIMESTAMP
from ehr2meds.preMEDS.dataloading import DataLoader
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ValueProcessor:
    def __init__(self, cfg) -> None:
        self.test = cfg.test
        logger.info(f"test {self.test}")
        self.data_loader = DataLoader(
            chunksize=cfg.data.chunksize,
            test=self.test,
        )
        self.numeric_value = cfg.data["numeric_value"]
        self.input_path = cfg.paths.input
        self.output_path = cfg.paths.output
        self.file_name = cfg.file_name

        self.normalization_type = None
        self.norm_params: Dict[str, dict] = {}
        if cfg.data.get("normalize"):
            self.normalization_type = cfg.data.normalize.norm_type
            self.build_distribution_on = cfg.data.normalize.build_distribution_on
            self.dist_path = cfg.data.normalize.dist_path

        self.aggregation_type = None
        self.agg_params: Dict[str, dict] = {}
        if cfg.data.get("aggregate"):
            self.aggregation_type = cfg.data.aggregate.agg_type

    def __call__(self):
        logger.info("Getting lab distribution")
        if self.dist_path:
            dist = pd.read_csv(self.dist_path)
        else:
            dist = self.get_lab_values(
                filename=self.file_name, 
                input_path=self.input_path,
                numeric_value=self.numeric_value,
                build_distribution_on=self.build_distribution_on,
                norm_type=self.normalization_type,
            )
        logger.info("Distribution data loaded")
        self.process_distribution_data(dist)
        logger.info("Processing data")
        self.process_data()

    def process_data(self):
        input_root = Path(self.input_path)
        output_path = Path(self.output_path)
        base_path = input_root / "data"
        for file_path in sorted(base_path.glob("*/*.parquet")):
            logger.info(f"Processing {file_path}")
            for chunk in tqdm(
                self.data_loader.load_chunks(filename=str(file_path)),
                desc=f"Processing {file_path}",
            ):
                if self.normalization_type:
                    chunk = self.normalize_chunk(chunk)
                if self.aggregation_type:
                    chunk = self.aggregate_chunk(chunk)
                rel_path = file_path.relative_to(base_path)
                print(chunk.head())
                self.save_chunk(chunk, rel_path, output_path)


    def process_distribution_data(self, dist: Dict[str, List[float]]) -> None:
        """Process distribution data based on normalization type."""
        if self.normalization_type == "min_max":
            self.process_minmax_distribution(dist)
        else:
            raise ValueError("Invalid type of normalization")

    def process_minmax_distribution(self, dist: Dict[str, List[float]]) -> None:
        """Process distribution data for min-max normalization."""
        self.norm_params["min_max"] = {
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
        if self.normalization_type == "min_max":
            return self.min_max_normalize(chunk, nums)
        raise ValueError(f"Invalid normalization type: {self.normalization_type}")
    
    def aggregate_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Aggregate rows with the same subject_id, code, and timestamp."""
        nan_rows = chunk[chunk[[TIMESTAMP]].isna().any(axis=1)]
        
        non_nan_rows = chunk.dropna(subset=[TIMESTAMP]).copy()

        if non_nan_rows.empty:
            out = chunk
        else:
            non_nan_rows[TIMESTAMP] = pd.to_datetime(
                non_nan_rows[TIMESTAMP], errors="coerce"
            )
            invalid_ts = non_nan_rows[TIMESTAMP].isna()
            if invalid_ts.any():
                nan_rows = pd.concat(
                    [nan_rows, non_nan_rows[invalid_ts]], ignore_index=True
                )
                non_nan_rows = non_nan_rows[~invalid_ts]

            if non_nan_rows.empty:
                out = nan_rows if not nan_rows.empty else chunk
            else:
                group_cols = [SUBJECT_ID, TIMESTAMP, CODE]
                grouped = non_nan_rows.groupby(group_cols, dropna=False)
                if self.aggregation_type == "list":
                    aggregated_df = grouped.first().reset_index()
                    aggregated_df[self.numeric_value] = grouped[
                        self.numeric_value
                    ].apply(list).tolist()
                else:
                    aggregated_df = grouped.agg(self.aggregation_type).reset_index()
                    aggregated_df[self.numeric_value] = [
                        [v] for v in aggregated_df[self.numeric_value]
                    ]

                parts = [aggregated_df]
                if not nan_rows.empty:
                    parts.append(nan_rows)
                out = pd.concat(parts, ignore_index=True)

        out = out.copy()
        out[self.numeric_value] = [
            v if isinstance(v, list) else [v] for v in out[self.numeric_value]
        ]
        return out

    def min_max_normalize(
        self, chunk: pd.DataFrame, nums: pd.Series
    ) -> pd.DataFrame:
        """Min-max scale numeric values for codes present in norm_params['min_max']."""
        codes = chunk[CODE]
        for code, (min_val, max_val) in self.norm_params["min_max"].items():
            mask = nums.notna() & (codes == code)
            if not mask.any() or max_val == min_val:
                continue
            scaled = (nums.loc[mask] - min_val) / (max_val - min_val)
            chunk.loc[mask, self.numeric_value] = np.clip(
                scaled, 0, 1
            ).to_numpy()
        return chunk