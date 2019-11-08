from typing import List, Dict, Any

import torch
from torch import Tensor
from torch import optim
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from agents import Agent
from utils import with_default_config, DataBatch, discount_rewards_to_go


class PPOTrainer:
    """
    A class that holds two (or a different number of) agents, and is responsible for performing the weight updates,
    using data collected by the evaluator (container here as well maybe?)

    The set of agents should not be changed. The state_dict should be alright to be loaded?
    """

    def __init__(self, agents: Dict[str, Agent], config: Dict[str, Any]):
        self.agents = agents
        self.agent_ids: List[str] = list(agents.keys())

        default_config = {
            "agents_to_optimize": None,
            "optimizer": optim.Adam,
            "optimizer_kwargs": {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "amsgrad": False
            },
            "gamma": 0.95,
            "eps": 0.1

        }
        self.config = with_default_config(config, default_config)
        self.agents_to_optimize: List[str] = self.agent_ids if self.config['agents_to_optimize'] is None \
            else self.config['agents_to_optimize']

        self.optimizers: Dict[str, Optimizer] = {
            agent_id: self.config["optimizer"](agent.model.parameters(), **self.config["optimizer_kwargs"])
            for agent_id, agent in self.agents.items() if agent_id in self.agents_to_optimize
        }

        self.gamma: float = self.config["gamma"]
        self.eps: float = self.config["eps"]

    def train_on_data(self, data_batch: DataBatch):
        for agent_id in self.agents_to_optimize:
            agent = self.agents[agent_id]
            optimizer = self.optimizers[agent_id]

            obs_batch = data_batch['observations'][agent_id]
            action_batch = data_batch['actions'][agent_id]
            reward_batch = data_batch['rewards'][agent_id]
            old_logprobs_batch = data_batch['logprobs'][agent_id]
            done_batch = data_batch['dones'][agent_id]
            state_batch = data_batch['states'][agent_id]

            logprob_batch, value_batch, entropy_batch = agent.evaluate_actions(obs_batch, action_batch, state_batch)

            discounted_batch = discount_rewards_to_go(reward_batch, done_batch, self.gamma)
            advantages_batch = (discounted_batch - value_batch).detach()

            # Compute the loss
            # prob_ratio = torch.exp(logprob_batch - old_logprobs_batch)
            # surr1 = prob_ratio * advantages_batch
            # surr2 = torch.clamp(prob_ratio, 1. - self.eps, 1 + self.eps) * advantages_batch
            #
            # policy_loss = -torch.min(surr1, surr2).mean()

            # compute PG first

            pg_loss = -logprob_batch * advantages_batch
            value_loss = (value_batch - discounted_batch)**2

            loss = (pg_loss + value_loss).mean()
            optimizer.zero_grad()

            loss.backward(retain_graph=True)
            optimizer.step()



if __name__ == '__main__':
    from rollout import Evaluator
    from models import MLPModel, LSTMModel
    from envs import foraging_env_creator

    env = foraging_env_creator({})

    agent_ids = ["Agent0", "Agent1"]
    agents_: Dict[str, Agent] = {
        agent_id: Agent(LSTMModel({}), name=agent_id)
        for agent_id in agent_ids
    }

    runner = Evaluator(agents_, env)
    data_batch = runner.rollout_steps(num_episodes=10, break_gradients=False, use_tqdm=True)
    obs_batch = data_batch['observations']['Agent0']
    action_batch = data_batch['actions']['Agent0']
    state_batch = data_batch['states']['Agent0']
    reward_batch = data_batch['rewards']['Agent0']
    done_batch = data_batch['dones']['Agent0']

    logprob_batch, value_batch, entropy_batch = agents_['Agent0'].evaluate_actions(obs_batch, action_batch, state_batch)
