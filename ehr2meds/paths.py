from pathlib import Path
import os


def get_config_path():
    return Path(os.getenv("EHR2MEDS_CONFIG_PATH"))


def get_data_path():
    return Path(os.getenv("EHR2MEDS_DATA"))
