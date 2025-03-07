import argparse
import logging
import pathlib
import shutil
from os.path import dirname, join, realpath

from ehr2meds.PREMEDS.azure_run.run import Run
from ehr2meds.PREMEDS.preprocessing.io.config import load_config
from ehr2meds.PREMEDS.preprocessing.premeds.extractor import PREMEDSExtractor

run = Run
run.name(f"MEDS")


def parse_args():
    parser = argparse.ArgumentParser(description="MEDS preprocessing script")
    parser.add_argument(
        "--config",
        type=str,
        default="MEDS",
        help="Name of the configuration file (without .yaml extension)",
    )
    return parser.parse_args()


def run_pre_MEDS(config_name):
    base_dir = dirname(realpath(__file__))
    config_path = join(base_dir, "configs")
    cfg = load_config(join(config_path, config_name + ".yaml"))
    pathlib.Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)  # added line
    shutil.copyfile(
        join(config_path, config_name + ".yaml"),
        join(cfg.paths.output_dir, "config.yaml"),
    )
    logging.basicConfig(
        filename=join(cfg.paths.output_dir, cfg.run_name + ".log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    extractor = PREMEDSExtractor(cfg, logger)
    extractor()
    return config_path, cfg


if __name__ == "__main__":
    args = parse_args()
    run_pre_MEDS(args.config)
