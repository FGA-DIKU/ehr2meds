import yaml


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
