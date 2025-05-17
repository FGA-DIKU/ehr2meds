import gzip
import os
import pickle
import shutil
from os.path import join

from ehr2meds.PREMEDS.preprocessing.extraction.extractor import ValueExtractor
from ehr2meds.PREMEDS.preprocessing.io.config import load_config
from ehr2meds.PREMEDS.preprocessing.io.logging import setup_logging


def my_app(config_path):
    """
    Run normalization with the given config file.

    :param config_path: Full path to the config file
    """
    # Load config directly from provided path
    cfg = load_config(config_path)

    # Create output directory (parent directory of the output file)
    output_dir = cfg.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Copy config to output directory
    shutil.copyfile(
        config_path,
        join(output_dir, "config.yaml"),
    )

    # Setup logging
    setup_logging(
        log_dir=cfg.get("logging", {}).get("path"),
        log_level=cfg.get("logging", {}).get("level"),
        name="extract_lab_dist.log",
    )

    lab_val_dict = ValueExtractor(cfg)()

    # Compress with gzip when saving
    with gzip.open(join(output_dir, "lab_val_dict.pickle.gz"), "wb") as f:
        pickle.dump(lab_val_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract lab distribution script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (.yaml)",
    )
    args = parser.parse_args()
    my_app(args.config)
