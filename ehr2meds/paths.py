import os


def get_config_path():
    return os.getenv("EHR2MEDS_CONFIGS")


def get_data_path():
    return os.getenv("EHR2MEDS_DATA")
