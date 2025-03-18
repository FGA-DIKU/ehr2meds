import argparse
import logging
import pathlib
import shutil
from os.path import join

from ehr2meds.PREMEDS.preprocessing.io.config import load_config
from ehr2meds.PREMEDS.preprocessing.premeds.extractor import PREMEDSExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="MEDS preprocessing script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (.yaml)",
    )
    return parser.parse_args()


def run_pre_MEDS(config_path):
    """
    Run PREMEDS preprocessing with the given config file.

    :param config_path: Full path to the config file
    """
    # Load config directly from provided path
    cfg = load_config(config_path)

    # Create output directory
    pathlib.Path(cfg.paths.output).mkdir(
        parents=True, exist_ok=True
    )  # changed to output instead of output_dir

    # Copy config to output directory
    shutil.copyfile(
        config_path,
        join(cfg.paths.output, "config.yaml"),
    )

    # Setup logging
    logging.basicConfig(
        filename=join(cfg.paths.output, cfg.run_name + ".log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    extractor = PREMEDSExtractor(cfg, logger)
    extractor()
    return cfg


if __name__ == "__main__":
    args = parse_args()
    run_pre_MEDS(args.config)
