from os.path import dirname, join, realpath
import shutil
from preprocessors.load import instantiate, load_config
import logging
from azure_run.run import Run
from azure_run import datastore

import pathlib

run = Run
run.name(f"norm_MEDS")
ds = datastore("workspaceblobstore")

config_name = "normalise"
def my_app(config_name):
    # datastore = Datastore.get(ws, 'workspaceblobstore')
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
    
    if cfg.env=='azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=join(cfg.paths.output_dir), datastore_name = cfg.data.data_store,
                    remote_path = cfg.data.save_path)
        logger.info('Finished')

if __name__=='__main__':
    my_app(config_name)