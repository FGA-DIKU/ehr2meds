import hydra
import pathlib
from dotenv import load_dotenv
from ehr2meds.paths import get_config_path
from ehr2meds.preMEDS.extractor import PREMEDSExtractor
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
    Run PREMEDS preprocessing with the given config file.

    :param config_path: Full path to the config file
    """
    # Create output directory
    print(cfg)

    pathlib.Path(cfg.paths.output).mkdir(parents=True, exist_ok=True)  # changed to output instead of output_dir

    extractor = PREMEDSExtractor(cfg)
    extractor()
    return cfg


if __name__ == "__main__":
    main()
