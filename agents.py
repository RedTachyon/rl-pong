import numpy as np

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical

from models import BaseModel, MLPModel

from typing import Tuple, Union, List


class Agent:
    def __init__(self, model: BaseModel, name: str):
        self.model = model
        self.name = name

        self.storage = {}

    def compute_actions(self, obs_batch: Tensor,
                        deterministic: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Computes the action for a batch of observations with given hidden states. Breaks gradients.

        Args:
            obs_batch: observation array in shape either (batch_size, obs_size)
            deterministic: boolean, whether to always take the best action

        Returns:
            action, logprob of the action, new state vectors
        """
        action_distribution: Categorical
        action_distribution, _ = self.model(obs_batch)
        if deterministic:
            actions = torch.argmax(action_distribution.probs, dim=-1)
        else:
            actions = action_distribution.sample()

        logprobs = action_distribution.log_prob(actions)

        return actions, logprobs

    def compute_single_action(self, obs: np.ndarray,
                              deterministic: bool = False) -> Tuple[int, float]:
        """
        Computes the action for a single observation with the given hidden state. Breaks gradients.

        Args:
            obs: observation array in shape either (obs_size) or (1, obs_size)
            deterministic: boolean, whether to always take the best action

        Returns:
            action, logprob of the action, new state vectors
        """
        obs = torch.tensor([obs])

        action, logprob = self.compute_actions(obs, deterministic)

        return action.item(), logprob.item()

    def evaluate_actions(self, obs_batch: Tensor,
                         action_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action, hidden_state)
        transitions. Preserves all the necessary gradients.

        Args:
            obs_batch: tensor of observations, (batch_size, obs_size)
            action_batch: tensor of actions, (batch_size, )

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """

        action_distribution, values = self.model(obs_batch)
        action_logprobs = action_distribution.log_prob(action_batch)
        entropies = action_distribution.entropy()

        return action_logprobs, values, entropies

    def get_initial_state(self):
        return self.model.get_initial_state()

    def reset(self):
        self.storage = {}



if __name__ == '__main__':
    pass
    # mlp_agent = Agent(MLPModel({}), "MLPAgent")
    # lstm_agent = Agent(LSTMModel({}), "LSTMAgent")
    #
    # env = foraging_env_creator({})
    # obs_ = env.reset()
    # obs_ = {k: torch.randn(3, 15) for k in obs_}
