from typing import Dict, List, TypeVar, Union, Tuple, Any

import torch
from torch import Tensor

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


def discount_rewards_to_go(rewards: Tensor, dones: Tensor, gamma: float = 1.):

    current_reward = 0
    discounted_rewards = []
    for reward, done in zip(rewards, dones):
        if done:
            current_reward = 0
        current_reward = reward + gamma * current_reward
        discounted_rewards.insert(0, current_reward)
    return torch.tensor(discounted_rewards)


DataBatch = Dict[str, Dict[str, Any]]

