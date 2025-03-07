import logging
import pathlib
import shutil
from os.path import dirname, join, realpath

from azure_run.run import Run
from preprocessors.load import instantiate, load_config

run = Run
run.name(f"MEDS")

config_name = "MEDS"
def run_pre_MEDS(config_name):
    base_dir = dirname(realpath(__file__))
    config_path = join(base_dir, 'configs')
    cfg = load_config(join(config_path, config_name+'.yaml'))
    pathlib.Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True) # added line
    shutil.copyfile(join(config_path, config_name+'.yaml'), join(cfg.paths.output_dir, 'config.yaml'))
    logging.basicConfig(filename=join(cfg.paths.output_dir, cfg.run_name+'.log'), level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    preprocessor = instantiate(cfg.preprocessor, {'cfg':cfg, 'logger':logger})
    preprocessor()
    return config_path, cfg

if __name__=='__main__':
    run_pre_MEDS(config_name)