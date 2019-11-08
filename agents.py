import numpy as np

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical

from models import BaseModel, MLPModel, LSTMModel

from typing import Tuple, Union, List

from envs import foraging_env_creator


class Agent:
    def __init__(self, model: BaseModel, name: str):
        self.model = model
        self.stateful = model.stateful
        self.name = name

    def compute_actions(self, obs_batch: Tensor,
                        state_batch: Tuple = (),
                        deterministic: bool = False) -> Tuple[Tensor, Tensor, Tuple]:
        """
        Computes the action for a batch of observations with given hidden states. Breaks gradients.

        Args:
            obs_batch: observation array in shape either (batch_size, obs_size)
            state_batch: tuple of state tensors of shape (batch_size, lstm_nodes)
            deterministic: boolean, whether to always take the best action

        Returns:
            action, logprob of the action, new state vectors
        """
        action_distribution: Categorical
        states: Tuple
        action_distribution, _, states = self.model(obs_batch, state_batch)
        if deterministic:
            actions = torch.argmax(action_distribution.probs, dim=-1)
        else:
            actions = action_distribution.sample()

        logprobs = action_distribution.log_prob(actions)

        return actions, logprobs, states

    def compute_single_action(self, obs: np.ndarray,  # TODO: consider returning the entropy here as well for metrics?
                              state: Tuple = (),
                              deterministic: bool = False) -> Tuple[int, float, Tuple]:
        """
        Computes the action for a single observation with the given hidden state. Breaks gradients.

        Args:
            obs: observation array in shape either (obs_size) or (1, obs_size)
            state: tuple of state tensors of shape (1, lstm_nodes)
            deterministic: boolean, whether to always take the best action

        Returns:
            action, logprob of the action, new state vectors
        """
        if len(obs.shape) == 1:
            obs = torch.tensor([obs])

        action, logprob, new_state = self.compute_actions(obs, state, deterministic)

        return action.item(), logprob.item(), new_state

    def evaluate_actions(self, obs_batch: Tensor,
                         action_batch: Tensor,
                         state_batch: Union[Tuple, List]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action, hidden_state)
        transitions. Preserves all the necessary gradients.

        Args:
            obs_batch: tensor of observations, (batch_size, obs_size)
            action_batch: tensor of actions, (batch_size, )
            state_batch: either a tuple of tensors (tensor(h1, h2, ...), tensor(c1, c2, ...))
                         or a list of such tuples [(tensor(h1), tensor(c1)), (tensor(h2), tensor(c2)))]
                         Tuple version is designed to work without BPTT
                         List version - with BPTT

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of
        """

        if isinstance(state_batch, Tuple):  # (tensor({h_i}), tensor({c_i})); BP
            action_distribution, values, states = self.model(obs_batch, state_batch)
            values = values.flatten()
            action_logprobs = action_distribution.log_prob(action_batch)
            entropies = action_distribution.entropy()
        elif isinstance(state_batch, List):  # List; BPTT
            action_logprobs = []
            values = []
            entropies = []
            for (obs, action, (h_, c_)) in zip(obs_batch, action_batch, state_batch):
                action_distribution, value, state = self.model(obs.view(1, -1), (h_.view(1, -1), c_.view(1, -1)))
                action_logprob = action_distribution.log_prob(action)
                entropy = action_distribution.entropy()

                action_logprobs.append(action_logprob)
                values.append(value.flatten())
                entropies.append(entropy)
            action_logprobs = torch.cat(action_logprobs)
            values = torch.cat(values)
            entropies = torch.cat(entropies)
        else:
            raise ValueError("state_batch should be either a tuple of tensors, or a list of such tuples")

        return action_logprobs, values, entropies

    def get_initial_state(self):
        return self.model.get_initial_state()


if __name__ == '__main__':
    mlp_agent = Agent(MLPModel({}), "MLPAgent")
    lstm_agent = Agent(LSTMModel({}), "LSTMAgent")

    # torch.tensor([1,2]).grad

    env = foraging_env_creator({})
    obs_ = env.reset()
    obs_ = {k: torch.randn(3, 15) for k in obs_}
