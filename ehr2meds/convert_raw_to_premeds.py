import hydra
from dotenv import load_dotenv
from ehr2meds.paths import get_config_path
from ehr2meds.preMEDS.extractor import PREMEDSExtractor
from omegaconf import DictConfig

load_dotenv()


@hydra.main(
    config_path=get_config_path(),
    config_name="root_config",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(cfg)
    extractor = PREMEDSExtractor(cfg)
    extractor()
    return cfg


if __name__ == "__main__":
    main()
