import logging
from os.path import dirname, split

import pandas as pd
from tqdm import tqdm

from ehr2meds.PREMEDS.preprocessing.constants import CODE
from ehr2meds.PREMEDS.preprocessing.io.dataloader import get_data_loader

logger = logging.getLogger(__name__)


class ValueExtractor:
    """
    The purpose is to extract the distribution of a given values.

    This class takes a path to a table.
    And an output folder.
    Iterates through the table, extracting two columns:
    <name> and <value> and creates a dictionary with the name as the key and the values as the values.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.test = cfg.test
        logger.info(f"test {self.test}")
        self._init_data_loader()
        self.numeric_value = cfg.data["numeric_value"]

    def _init_data_loader(self):
        self.data_loader = get_data_loader(
            path=dirname(self.cfg.paths.input),
            env=self.cfg.env,
            chunksize=self.cfg.data.chunksize,
            test=self.test,
            test_rows=self.cfg.data.get("test_rows", 100_000),
        )

    def __call__(self):
        print("Getting lab distribution")
        dist = self.get_lab_values()
        return dist

    def get_lab_values(self):
        logger.info("Getting lab distribution")
        lab_val_dict = {}
        counter = 0

        for chunk in tqdm(
            self.data_loader.load_chunks(
                filename=split(self.cfg.paths.input)[1], cols=[self.numeric_value, CODE]
            ),
            desc="Building lab distribution",
        ):
            if self.numeric_value not in chunk.columns or CODE not in chunk.columns:
                raise ValueError(
                    f"Missing required columns. Available columns: {chunk.columns}"
                )
            logger.info(f"Loaded {self.cfg.data.chunksize*counter}")
            chunk[self.numeric_value] = pd.to_numeric(
                chunk[self.numeric_value], errors="coerce"
            )
            chunk = chunk.dropna(subset=[self.numeric_value])
            grouped = chunk.groupby(CODE)[self.numeric_value].apply(list).to_dict()

            for key, values in grouped.items():
                if key in lab_val_dict:
                    lab_val_dict[key].extend(values)
                else:
                    lab_val_dict[key] = values

            counter += 1
        return lab_val_dict
