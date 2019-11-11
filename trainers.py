from typing import List, Dict, Any, Union, Optional

import torch
from torch import Tensor
from torch import optim
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

import gym

from agents import Agent
from envs import MultiAgentEnv
from utils import with_default_config, DataBatch, discount_rewards_to_go, Timer

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from datetime import datetime
from tqdm import trange


class PPOTrainer:
    """
    A class that holds two (or a different number of) agents, and is responsible for performing the weight updates,
    using data collected by the evaluator (container here as well maybe?)

    The set of agents should not be changed. The state_dict should be alright to be loaded?
    """

    def __init__(self, agents: Dict[str, Agent], env: gym.Env, config: Dict[str, Any]):
        self.agents = agents
        self.agent_ids: List[str] = list(agents.keys())

        self.env = env

        default_config = {
            "agents_to_optimize": None,
            "batch_size": 10000,  # Number of steps to sample at each iteration, TODO: make it possible to use epochs
            "optimizer": optim.Adam,
            "optimizer_kwargs": {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "amsgrad": False
            },
            "gamma": 0.95,  # Discount factor
            "eps": 0.1,     # PPO clip parameter
            "tensorboard_name": "test"

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

        self.writer: SummaryWriter
        if self.config["tensorboard_name"]:
            dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = Path.home() / "tb_logs" / f"{self.config['tensorboard_name']}_{dt_string}"
            self.writer = SummaryWriter(str(path))
        else:
            self.writer = None

        self.collector = Collector(agents=self.agents, env=self.env)

    def train_on_data(self, data_batch: DataBatch,
                      step: int = 0,
                      extra_metrics: Optional[Dict[str, Any]] = None,
                      timer: Optional[Timer] = None):
        """
        Performs a single update step with PPO (WiP) on the given batch of data.

        Args:
            data_batch: DataBatch, dictionary
            step:
            extra_metrics:
            timer:

        Returns:

        """
        metrics = {}
        if timer is None:
            timer = Timer()
        for agent_id in self.agents_to_optimize:
            agent = self.agents[agent_id]
            optimizer = self.optimizers[agent_id]

            ####################################### Unpack and prepare the data #######################################
            obs_batch = data_batch['observations'][agent_id]
            action_batch = data_batch['actions'][agent_id]
            reward_batch = data_batch['rewards'][agent_id]
            old_logprobs_batch = data_batch['logprobs'][agent_id]
            done_batch = data_batch['dones'][agent_id]
            # state_batch = data_batch['states'][agent_id]  # unused

            timer.checkpoint()
            logprob_batch, value_batch, entropy_batch = agent.evaluate_actions(obs_batch, action_batch, done_batch)

            metrics[f"{agent_id}/rollout_eval_time"] = timer.checkpoint()

            discounted_batch = discount_rewards_to_go(reward_batch, done_batch, self.gamma)
            advantages_batch = (discounted_batch - value_batch).detach()
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-6)

            ############################################# Compute the loss #############################################
            # prob_ratio = torch.exp(logprob_batch - old_logprobs_batch)
            # surr1 = prob_ratio * advantages_batch
            # surr2 = torch.clamp(prob_ratio, 1. - self.eps, 1 + self.eps) * advantages_batch
            #
            # policy_loss = -torch.min(surr1, surr2).mean()

            ### PG loss ###
            pg_loss = -logprob_batch * advantages_batch
            value_loss = (value_batch - discounted_batch)**2

            loss = (pg_loss + value_loss).mean()

            ############################################### Update step ###############################################
            timer.checkpoint()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            metrics[f"{agent_id}/update_time"] = timer.checkpoint()

            ############################################# Collect metrics ############################################

            # Delay by one, so that the new episode starts after a done=True, with a 0 at the beginning
            episode_indices = done_batch.cumsum(dim=0)[:-1]
            episode_indices = torch.cat([torch.tensor([0]), episode_indices])  # [0, 0, 0, ..., 1, 1, ..., 2, ..., ...]

            ep_ids, ep_lens_tensor = torch.unique(episode_indices, return_counts=True)
            ep_lens = tuple(ep_lens_tensor)  # tuple of episode lengths

            # Group rewards by episode and sum them up
            ep_rewards = torch.tensor([torch.sum(rewards) for rewards in torch.split(reward_batch, ep_lens)])

            ### Add new training-based metrics here ###
            metrics[f"{agent_id}/episode_len_mean"]      = torch.mean(ep_lens_tensor.float()).item()
            metrics[f"{agent_id}/episode_reward_mean"]   = torch.mean(ep_rewards).item()
            metrics[f"{agent_id}/episode_reward_median"] = torch.median(ep_rewards).item()
            metrics[f"{agent_id}/episode_reward_min"]    = torch.min(ep_rewards).item()
            metrics[f"{agent_id}/episode_reward_max"]    = torch.max(ep_rewards).item()
            metrics[f"{agent_id}/episode_reward_std"]    = torch.std(ep_rewards).item()
            metrics[f"{agent_id}/episodes_this_iter"]    = len(ep_ids)
            metrics[f"{agent_id}/mean_entropy"]          = torch.mean(entropy_batch).item()

            if extra_metrics is not None:
                metrics = with_default_config(metrics, extra_metrics)  # add extra_metrics if not computed here

            self.write_dict(metrics, step)

    def write_dict(self, metrics: Dict[str, Union[int, float]], step: int):
        if self.writer is not None:
            self.writer: SummaryWriter
            for key, value in metrics.items():
                self.writer.add_scalar(tag=key, scalar_value=value, global_step=step)

    def train(self, num_iterations: int,
              starting_iteration: int = 0,
              disable_tqdm: bool = False):

        timer = Timer()
        for step in trange(starting_iteration, starting_iteration + num_iterations, disable=disable_tqdm):
            timer.checkpoint()
            data_batch = self.collector.collect_data(num_steps=self.config["batch_size"])
            data_time = timer.checkpoint()
            time_metric = {f"{agent_id}/data_collection_time": data_time for agent_id in self.agent_ids}

            self.train_on_data(data_batch, step, extra_metrics=time_metric, timer=timer)



if __name__ == '__main__':
    from rollout import Collector
    from models import MLPModel, LSTMModel
    from envs import foraging_env_creator

    # env_ = foraging_env_creator({})

    # agent_ids = ["Agent0", "Agent1"]
    # agents_: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents_, env_)
    # data_batch = runner.rollout_steps(num_episodes=10, disable_tqdm=True)
    # obs_batch = data_batch['observations']['Agent0']
    # action_batch = data_batch['actions']['Agent0']
    # reward_batch = data_batch['rewards']['Agent0']
    # done_batch = data_batch['dones']['Agent0']
    #
    # logprob_batch, value_batch, entropy_batch = agents_['Agent0'].evaluate_actions(obs_batch, action_batch, done_batch)
