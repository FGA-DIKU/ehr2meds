from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig

from ehr_preprocess.processors import utils, processors

config_name = "config"
base_dir = dirname(dirname(realpath(__file__)))
config_path = join(base_dir, 'configs')

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    processor = hydra.utils.instantiate(cfg.processor, cfg=cfg)
    #if cfg.convert_csv_to_parquet:
     #   csv_to_parquet = utils.MIMIC_CSV_to_Parquet_Converter(cfg, test=False)
      #  csv_to_parquet()
    # hydra.utils.instantiate(cfg)
    

if __name__=='__main__':
    my_app()