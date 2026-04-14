import hydra
import pathlib
from dotenv import load_dotenv
from ehr2meds.paths import get_config_path
from ehr2meds.preMEDS.normalizer import Normalizer
from ehr2meds.preMEDS.logging import setup_logging
from omegaconf import DictConfig, OmegaConf
from os.path import join

load_dotenv()


@hydra.main(
    config_path=get_config_path(),
    config_name="root_config",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    """
    Run normalization on preMEDS data with the given config file.

    :param config_path: Full path to the config file
    """

    print(cfg)
    # Create output directory
    pathlib.Path(cfg.paths.output).mkdir(
        parents=True, exist_ok=True
    )  # changed to output instead of output_dir

    # Copy config to output directory
    OmegaConf.save(cfg, join(cfg.paths.output, "config.yaml"))
    print(cfg)
    setup_logging(
        log_dir=cfg["logging"]["path"],
        log_level=cfg["logging"]["level"],
        name="preMEDS.log",
    )


    normalizer = Normalizer(cfg)
    normalizer()
    return cfg


if __name__ == "__main__":
    main()
