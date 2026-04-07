import argparse
import pathlib
import shutil
from os.path import join
from pathlib import Path


from ehr2meds.preMEDS.config import load_config
from ehr2meds.preMEDS.extractor import PREMEDSExtractor
from ehr2meds.preMEDS.logging import setup_logging


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
    root = Path("ehr2meds")
    config_path = root / "configs" / config_path
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

    setup_logging(
        log_dir=cfg.get("logging", {}).get("path"),
        log_level=cfg.get("logging", {}).get("level"),
        name="preMEDS.log",
    )

    extractor = PREMEDSExtractor(cfg)
    extractor()
    return cfg


if __name__ == "__main__":
    args = parse_args()
    run_pre_MEDS(args.config)
