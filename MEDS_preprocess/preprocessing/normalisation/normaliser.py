import pickle
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from MEDS_preprocess.preprocessing.io.azure import get_data_loader


class Normaliser:
    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.test = cfg.test
        self.logger.info(f"test {self.test}")
        self.firstRound = True
        self.normalisation_type = cfg.data["norm_type"]

        # Initialize data loader based on environment
        self.data_loader = get_data_loader(
            env=cfg.env,
            datastore=cfg.data.get("data_store"),
            dump_path=cfg.data.get("data_dir"),
            logger=logger,
        )

        # Load distribution data
        if "dist_path" not in cfg.data:
            dist = self.get_lab_dist()
            dist_save_path = join(cfg.paths.output_dir, "lab_val_dict.pkl")
            with open(dist_save_path, "wb") as f:
                pickle.dump(dist, f)
            self.logger.info(f"Saved lab distribution to {dist_save_path}")
        else:
            with open(join(cfg.data.dist_path, "lab_val_dict.pkl"), "rb") as f:
                dist = pickle.load(f)
            with open(join(cfg.data.dist_path, "vocabulary.pkl"), "rb") as f:
                self.vocab = pickle.load(f)

        # Process distribution data based on normalization type
        if self.normalisation_type == "Min_max":
            self.min_max_vals = {
                concept: (
                    (
                        np.percentile(dist[concept], 0.01 * 100)
                        if len(dist[concept]) > 1
                        else dist[concept][0]
                    ),
                    (
                        np.percentile(dist[concept], 0.99 * 100)
                        if len(dist[concept]) > 1
                        else dist[concept][0]
                    ),
                )
                for concept in dist
                if dist[concept]
            }
        elif self.normalisation_type == "Categorise":
            self.quantiles = {
                concept: (
                    np.percentile(sorted(dist[concept]), [25, 50, 75])
                    if len(dist[concept]) > 0
                    else (0, 0, 0)
                )
                for concept in dist
            }
        elif self.normalisation_type == "Quantiles":
            self.n_quantiles = cfg.data["n_quantiles"]
            self.quantiles = {
                concept: (
                    [
                        np.percentile(sorted(dist[concept]), i)
                        for i in np.linspace(
                            100 / self.n_quantiles, 100, self.n_quantiles
                        )
                    ]
                    if len(dist[concept]) > 0
                    else [0] * self.n_quantiles
                )
                for concept in dist
            }
        else:
            raise ValueError("Invalid type of normalisation")

    def __call__(self):
        cfg = self.cfg
        save_name = cfg.data.save_name
        if not Path(join(cfg.paths.output_dir, save_name)).exists():
            counter = 0
            # Iterate over chunks of the CSV file
            for chunk in tqdm(
                self.data_loader.load_chunks(
                    cfg.data.filename, chunk_size=cfg.data.chunksize, test=self.test
                ),
                desc="Processing chunks",
            ):
                if "Column1" in chunk.columns:
                    chunk = chunk.drop(columns="Column1")
                chunk = chunk.reset_index(drop=True)
                self.logger.info(f"Loaded {cfg.data.chunksize*counter}")
                chunk_processed = self.process_chunk(chunk)

                # Save processed chunk
                output_path = join(cfg.paths.output_dir, f"concept.{save_name}")
                mode = "w" if counter == 0 else "a"
                if output_path.endswith(".parquet"):
                    chunk_processed.to_parquet(output_path, index=False, mode=mode)
                else:
                    chunk_processed.to_csv(output_path, index=False, mode=mode)

                counter += 1

    def get_lab_dist(self):
        self.logger.info("Getting lab distribution")
        cfg = self.cfg
        lab_val_dict = {}
        counter = 0

        for chunk in tqdm(
            self.data_loader.load_chunks(
                cfg.data.filename, chunk_size=cfg.data.chunksize, test=self.test
            ),
            desc="Building lab distribution",
        ):
            self.logger.info(f"Loaded {cfg.data.chunksize*counter}")
            chunk["numeric_value"] = pd.to_numeric(
                chunk["numeric_value"], errors="coerce"
            )
            chunk = chunk.dropna(subset=["numeric_value"])
            grouped = chunk.groupby("code")["numeric_value"].apply(list).to_dict()

            for key, values in grouped.items():
                if key in lab_val_dict:
                    lab_val_dict[key].extend(values)
                else:
                    lab_val_dict[key] = values

            counter += 1
        return lab_val_dict

    def process_chunk(self, chunk):
        chunk["numeric_value"] = chunk.apply(self.normalise, axis=1)
        return chunk

    def normalise(self, row):
        concept = row["code"]
        value = row["numeric_value"]
        if not pd.notnull(pd.to_numeric(value, errors="coerce")):
            return value
        else:
            value = pd.to_numeric(value)

        if self.normalisation_type == "Min_max":
            return self.min_max_normalise(concept, value)
        elif self.normalisation_type == "Quantiles":
            return self.quantile(concept, value)
        else:
            Warning(f"Normalisation type {self.normalisation_type} not implemented")

    def min_max_normalise(self, concept, value):
        if concept in self.min_max_vals:
            (min_val, max_val) = self.min_max_vals[concept]
            if max_val != min_val:
                normalised_value = (value - min_val) / (max_val - min_val)
                return round(max(0, min(1, normalised_value)), 3)
            else:
                return "UNIQUE"
        else:
            return "N/A"

    def quantile(self, concept, value):
        if concept not in self.quantiles:
            return "N/A"
        else:
            quantile_values = self.quantiles[concept]
            if len(quantile_values) != self.n_quantiles:
                raise ValueError(
                    f"Expected {self.n_quantiles} quantiles for concept '{concept}'"
                )
            for i, q in enumerate(quantile_values, start=1):
                if value <= q:
                    return f"Q{i}"
            return f"Q{self.n_quantiles}"
