from typing import Dict


def with_default_config(config: Dict, default: Dict):
    config = config.copy()
    for key in default.keys():
        config.setdefault(key, default[key])
    return config

