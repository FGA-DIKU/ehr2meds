import yaml
from dataclasses import dataclass
from typing import Dict

from ehr2meds.PREMEDS.preprocessing.constants import ADMISSION_IND


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg = Config(cfg)
    return cfg


class Config(dict):
    def __init__(self, dictionary=None):
        super(Config, self).__init__()
        if dictionary:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = Config(value)
                self[key] = value
                setattr(self, key, value)

    def __setattr__(self, key, value):
        super(Config, self).__setattr__(key, value)
        super(Config, self).__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        super(Config, self).__setattr__(key, value)

    def __delattr__(self, name):
        if name in self:
            del self[name]
        if hasattr(self, name):
            super(Config, self).__delattr__(name)

    def __delitem__(self, name):
        if name in self:
            del self[name]
        if hasattr(self, name):
            super(Config, self).__delattr__(name)


@dataclass
class AdmissionsConfig:
    """Configuration for admissions processing."""

    type_column: str = "type"
    section_column: str = "section"
    timestamp_in_column: str = "timestamp_in"
    timestamp_out_column: str = "timestamp_out"
    admission_event_type: str = ADMISSION_IND.lower()
    transfer_event_type: str = "flyt ind"
    use_adm_move: bool = False
    save_adm_move: bool = False
    rename_columns: Dict[str, str] = None
    filename: str = "admissions"
    prefix: str = None

    def __post_init__(self):
        if self.rename_columns is None:
            self.rename_columns = {}
