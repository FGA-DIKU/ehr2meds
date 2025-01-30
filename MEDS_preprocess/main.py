from os.path import dirname, join, realpath
import shutil
from preprocessors.load import instantiate, load_config
import logging
from azure_run.run import Run
from azure_run import datastore
from azureml.core import Dataset

import pathlib
import subprocess

run = Run
run.name(f"MEDS")
ds_sp = datastore("researcher_data")
blob_ds = datastore("workspaceblobstore")

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
    preprocessor = instantiate(cfg.preprocessor, {'cfg':cfg, 'logger':logger, 'datastore':ds_sp, 'dump_path': cfg.paths.dump_path, 'file_datastore':blob_ds})
    preprocessor()
    return config_path, cfg

if __name__=='__main__':
    run_pre_MEDS(config_name)