import pathlib
import shutil
from os.path import join, dirname

from ehr2meds.PREMEDS.preprocessing.io.config import load_config
from ehr2meds.PREMEDS.preprocessing.normalisation.normaliser import Normaliser
from ehr2meds.PREMEDS.preprocessing.io.logging import setup_logging


def my_app(config_path):
    """
    Run normalization with the given config file.

    :param config_path: Full path to the config file
    """
    # Load config directly from provided path
    cfg = load_config(config_path)

    # Create output directory (parent directory of the output file)
    output_dir = dirname(cfg.paths.output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Copy config to output directory
    shutil.copyfile(
        config_path,
        join(output_dir, "config.yaml"),
    )

    # Setup logging
    setup_logging(
        log_dir=cfg.get("logging", {}).get("path"),
        log_level=cfg.get("logging", {}).get("level"),
        name="normalise.log",
    )

    preprocessor = Normaliser(cfg)
    preprocessor()
    return cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalization script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (.yaml)",
    )
    args = parser.parse_args()
    my_app(args.config)
