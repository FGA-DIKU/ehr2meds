import logging
import pathlib
import shutil
from os.path import join, dirname

from ehr2meds.PREMEDS.preprocessing.io.config import load_config
from ehr2meds.PREMEDS.preprocessing.normalisation.normaliser import Normaliser


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
    logging.basicConfig(
        filename=join(output_dir, cfg.run_name + ".log"),
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    preprocessor = Normaliser(cfg, logger)
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
