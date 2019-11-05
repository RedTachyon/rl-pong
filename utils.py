from typing import Dict, List, TypeVar

T = TypeVar('T')


def with_default_config(config: Dict, default: Dict) -> Dict:
    """
    Adds keys from default to the config, if they don't exist there yet.
    Serves to ensure that all necessary keys are always present.

    Args:
        config: config dictionary
        default: config dictionary with default values

    Returns:
        config with the defaults added
    """
    config = config.copy()
    for key in default.keys():
        config.setdefault(key, default[key])
    return config


def append_dict(var: Dict[str, T], data_dict: Dict[str, List[T]]):
    """
    Works like append, but operates on dictionaries of lists and dictionaries of values (as opposed to lists and values)

    Args:
        var: values to be appended
        data_dict: lists to be appended to
    """
    for key, value in var.items():
        data_dict[key].append(value)
