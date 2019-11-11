from typing import Dict, List, TypeVar, Union, Tuple, Any

import torch
from torch import Tensor
import time


DataBatch = Dict[str, Dict[str, Any]]


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


T = TypeVar('T')


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
    """
    Computes the discounted rewards to go, handling episode endings. Nothing unusual.
    """
    dones = dones.to(torch.int32)  # Torch can't handle reversing boolean tensors
    current_reward = 0
    discounted_rewards = []
    for reward, done in zip(rewards.flip(0), dones.flip(0)):
        if done:
            current_reward = 0
        current_reward = reward + gamma * current_reward
        discounted_rewards.insert(0, current_reward)
    return torch.tensor(discounted_rewards)


class Timer:
    """
    Simple timer to
    """
    def __init__(self):
        self.start = time.time()

    def checkpoint(self):
        now = time.time()
        diff = now - self.start
        self.start = now
        return diff


