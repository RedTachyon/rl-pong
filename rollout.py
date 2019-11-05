import numpy as np

from typing import Dict, Callable, Union, List, Tuple

from agents import Agent
from envs import MultiAgentEnv

import torch
from torch import Tensor

from utils import append_dict


class Memory:
    """
    Holds the rollout data in a nested dictionary structure as follows:
    {
        "observations":
            {
                "Agent0": [obs1, obs2, ...],
                "Agent1": [obs1, obs2, ...]
            },
        "actions":
            {
                "Agent0": [act1, act2, ...],
                "Agent1": [act1, act2, ...]
            },
        ...,
        "states":
            {
                "Agent0": [(h1, c1), (h2, c2), ...]
                "Agent1": [(h1, c1), (h2, c2), ...]
            }
    }
    """

    def __init__(self, agents: List[str]):
        """
        Creates the memory container. The only argument is a list of agent names to set up the dictionaries.

        Args:
            agents: names of agents
        """

        self.agents = agents

        _observations: Dict[str, List[np.ndarray]] = {agent: [] for agent in self.agents}
        _actions: Dict[str, List[int]] = {agent: [] for agent in self.agents}
        _rewards: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        _logprobs: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        _dones = Dict[str, List[bool]] = {agent: [] for agent in self.agents}
        _states = Dict[str, List[Tuple[Tensor, Tensor]]] = {agent: [] for agent in self.agents}

        self.data = {
            "observations": _observations,
            "actions": _actions,
            "rewards": _rewards,
            "logprobs": _logprobs,
            "dones": _dones,
            "states": _states,
        }

    def store(self,
              obs: Dict[str, np.ndarray],
              action: Dict[str, int],
              reward: Dict[str, float],
              logprob: Dict[str, float],
              done: Dict[str, bool],
              state: Dict[str, Tuple[Tensor, Tensor]]):

        update = (obs, action, reward, logprob, done, state)
        for key, var in zip(self.data, update):
            append_dict(var, self.data[key])

    def reset(self):
        for key in self.data:
            self.data[key] = {agent: [] for agent in self.agents}

    def get_optimized_data(self):
        return {
            "observations": np.stack(self.data["observations"]),  # (batch_size, obs_size) float
            "actions": np.array(self.data["actions"]),            # (batch_size, ) int
            "rewards": np.array(self.data["rewards"]),            # (batch_size, ) float
            "logprobs": np.array(self.data["logprobs"]),          # (batch_size, ) float
            "dones": np.array(self.data["dones"]),                # (batch_size, ) bool
            "states": 0 # TODO optimize this
        }


class Evaluator:
    def __init__(self, agents: Dict[str, Agent], env: MultiAgentEnv):
        self.agents = agents
        self.env = env

    def rollout(self, deterministic: Dict[str, bool]):
        obs = self.env.reset()
        obs_batch = [obs]

        done = False
        while not done:
            pass  # TODO: Put some thought in bookkeeping here

    def get_actions(self, obs: Union[Tensor, np.ndarray]) -> Dict[str, int]:
        pass
