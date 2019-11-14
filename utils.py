from typing import Dict, List, TypeVar, Union, Tuple, Any, Callable, Type

import torch
from torch import Tensor
import torch.nn.functional as F
import time

from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamw import AdamW
from torch.optim.adamax import Adamax
from torch.optim.sgd import SGD


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


def discount_rewards_to_go(rewards: Tensor, dones: Tensor, gamma: float = 1.) -> Tensor:
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


def get_optimizer(opt_name: str) -> Type[Optimizer]:
    optimizers = {
        "adam": Adam,
        "adadelta": Adadelta,
        "adamw": AdamW,
        "adagrad": Adagrad,
        "adamax": Adamax,
        "sgd": SGD
    }

    if opt_name not in optimizers.keys():
        raise ValueError(f"Wrong optimizer: {opt_name} is not a valid optimizer name. ")

    return optimizers[opt_name]


def get_activation(act_name: str) -> Callable[[Tensor], Tensor]:
    activations = {
        "relu": F.relu,
        "relu6": F.relu6,
        "elu": F.elu,
        "leaky_relu": F.leaky_relu,
        "sigmoid": F.sigmoid,
        "tanh": F.tanh,
        "softmax": F.softmax,
    }

    if act_name not in activations.keys():
        raise ValueError(f"Wrong activation: {act_name} is not a valid activation function name.")

    return activations[act_name]


class Timer:
    """
    Simple timer to
    """
    def __init__(self):
        self.start = time.time()

    def checkpoint(self) -> float:
        now = time.time()
        diff = now - self.start
        self.start = now
        return diff

def convert(elements: Tuple, names: List[str]):
    pass