import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical

from models import BaseModel, MLPModel, LSTMModel

from typing import Tuple

from envs import foraging_env_creator


class Agent:
    def __init__(self, model: BaseModel, name: str):
        self.model = model
        self.stateful = model.stateful
        self.name = name

    def compute_actions(self, obs_batch: Tensor,
                        state_batch: Tuple = (),
                        deterministic: bool = False) -> Tuple[Tensor, Tensor, Tuple]:

        action_distribution: Categorical
        states: Tuple
        action_distribution, _, states = self.model(obs_batch, state_batch)
        if deterministic:
            actions = torch.argmax(action_distribution.probs, dim=-1)
        else:
            actions = action_distribution.sample()

        logprobs = action_distribution.log_prob(actions)

        return actions, logprobs, states

    def compute_single_action(self, obs: Tensor,
                              state: Tuple = (),
                              deterministic: bool = False) -> Tuple[int, float, Tuple]:
        action, logprob, new_state = self.compute_actions(obs, state, deterministic)

        return action.item(), logprob.item(), new_state

    def evaluate_actions(self, obs_batch: Tensor, action_batch: Tensor, state_batch: Tuple):

        action_distribution: Categorical
        values: Tensor
        states: Tensor
        action_distribution, values, states = self.model(obs_batch, state_batch)

        action_logprobs = action_distribution.log_prob(action_batch)
        entropies = action_distribution.entropy()

        return action_logprobs, values, entropies


if __name__ == '__main__':
    mlp_agent = Agent(MLPModel({}), "MLPAgent")
    lstm_agent = Agent(LSTMModel({}), "LSTMAgent")

    env = foraging_env_creator({})
    obs_ = env.reset()
    obs_ = {k: torch.tensor(obs_[k]) for k in obs_}
