from typing import Dict, Tuple

from torch import nn as nn, Tensor
from torch.distributions import Distribution, Categorical
from torch.nn import functional as F

from envs import foraging_env_creator


env_config = {
    "rows": 7,
    "cols": 7,
    "subgoals": 2,
    "random_positions": True,
    "max_steps": 100
}

env = foraging_env_creator(env_config)


