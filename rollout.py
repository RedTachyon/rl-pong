import gym
import numpy as np

from typing import Dict, Callable, List, Tuple, Optional, Union

from agents import Agent
from envs import MultiAgentEnv, VectorizedEnvSP

import torch
from torch import Tensor

from utils import append_dict, DataBatch, convert_obs_to_dict, convert_action_to_env, preprocess_frame

from tqdm import trange

from collections import defaultdict


# See the Memory docstring for an example

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
        _dones: Dict[str, List[bool]] = {agent: [] for agent in self.agents}

        self.data = {
            "observations": _observations,
            "actions": _actions,
            "rewards": _rewards,
            "logprobs": _logprobs,
            "dones": _dones,
        }

        self.torch_data = {}

    def store(self,
              obs: Dict[str, np.ndarray],
              action: Dict[str, int],
              reward: Dict[str, float],
              logprob: Dict[str, float],
              done: Dict[str, bool]):

        update = (obs, action, reward, logprob, done)
        for key, var in zip(self.data, update):
            append_dict(var, self.data[key])

    def reset(self):
        for key in self.data:
            self.data[key] = {agent: [] for agent in self.agents}

        self.torch_data = {}

    def apply_to_agent(self, func: Callable):
        return {
            agent: func(agent) for agent in self.agents
        }

    def get_torch_data(self):
        observations = self.apply_to_agent(lambda agent: torch.tensor(np.stack(self.data["observations"][agent])))
        actions = self.apply_to_agent(lambda agent: torch.tensor(self.data["actions"][agent]))
        rewards = self.apply_to_agent(lambda agent: torch.tensor(self.data["rewards"][agent]))
        logprobs = self.apply_to_agent(lambda agent: torch.tensor(self.data["logprobs"][agent]))
        dones = self.apply_to_agent(lambda agent: torch.tensor(self.data["dones"][agent]))

        self.torch_data = {
            "observations": observations,  # (batch_size, obs_size) float
            "actions": actions,  # (batch_size, ) int
            "rewards": rewards,  # (batch_size, ) float
            "logprobs": logprobs,  # (batch_size, ) float
            "dones": dones,  # (batch_size, ) bool
        }

        return self.torch_data

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return self.data.__str__()


class Collector:
    """
    Class to perform data collection from two agents.
    """

    def __init__(self, agents: Dict[str, Agent], env: gym.Env, tuple_mode: bool = False):
        self.agents = agents
        self.agent_ids: List[str] = list(self.agents.keys())
        self.env = env
        self.memory = Memory(self.agent_ids)

        self.tuple_mode = tuple_mode

    def collect_data(self,
                     num_steps: Optional[int] = None,  # TODO: handle episode ends?
                     num_episodes: Optional[int] = None,
                     deterministic: Optional[Dict[str, bool]] = None,
                     disable_tqdm: bool = True,
                     max_steps: int = 102,
                     reset_memory: bool = True,
                     include_last: bool = False,
                     finish_episode: bool = True,
                     divide_rewards: Optional[int] = None,
                     visual: bool = False) -> DataBatch:

        """
        Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

        Args:
            num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
            num_episodes: number of episodes to generate
            deterministic: whether each agent should use the greedy policy; False by default
            disable_tqdm: whether a live progress bar should be (not) displayed
            max_steps: maximum number of steps that can be taken in episodic mode, recommended just above env maximum
            reset_memory: whether to reset the memory before generating data
            include_last: whether to include the last observation in episodic mode - useful for visualizations
            finish_episode: in step mode, whether to finish the last episode (resulting in more steps than num_steps)

        Returns: dictionary with the gathered data in the following format:

        {
            "observations":
                {
                    "Agent0": tensor([obs1, obs2, ...]),

                    "Agent1": tensor([obs1, obs2, ...])
                },
            "actions":
                {
                    "Agent0": tensor([act1, act2, ...]),

                    "Agent1": tensor([act1, act2, ...])
                },
            ...,


        }
        """
        assert not ((num_steps is None) == (num_episodes is None)), ValueError("Exactly one of num_steps, num_episodes "
                                                                               "should receive a value")

        if deterministic is None:
            deterministic = {agent_id: False for agent_id in self.agent_ids}

        if reset_memory:
            self.reset()

        # obs: Union[Tuple, Dict]
        obs = self.env.reset()

        if self.tuple_mode:  # Convert obs to dict
            obs = convert_obs_to_dict(obs, self.agent_ids)

        obs = {key: preprocess_frame(obs_) for key, obs_ in obs.items()}

        episode = 0

        for agent_id, agent in self.agents.items():
            agent.storage["last_obs"] = obs[agent_id]

        end_flag = False
        full_steps = (num_steps + 100 * int(finish_episode)) if num_steps else max_steps * num_episodes
        for step in trange(full_steps, disable=disable_tqdm):
            # Compute the action for each agent

            stacked_obs = {}
            for agent_id, agent in self.agents.items():
                stacked_obs[agent_id] = np.stack([obs[agent_id], agent.storage.get("last_obs")], axis=0)

            # breakpoint()
            action_info = {  # action, logprob
                agent_id: self.agents[agent_id].compute_single_action(stacked_obs[agent_id],
                                                                      deterministic[agent_id])
                for agent_id in self.agent_ids
            }

            # Unpack the actions
            action = {agent_id: action_info[agent_id][0] for agent_id in self.agent_ids}
            logprob = {agent_id: action_info[agent_id][1] for agent_id in self.agent_ids}

            # Actual step in the environment

            if self.tuple_mode:  # Convert action to env-compatible
                env_action = convert_action_to_env(action, self.agent_ids)
            else:
                env_action = action

            next_obs, reward, done, info = self.env.step(env_action)

            if self.tuple_mode:  # Convert outputs to dicts
                next_obs = convert_obs_to_dict(next_obs, self.agent_ids)
                reward = convert_obs_to_dict(reward, self.agent_ids)
                done = {agent_id: done for agent_id in self.agent_ids}

            next_obs = {key: preprocess_frame(obs_) for key, obs_ in next_obs.items()}

            if divide_rewards:
                reward = {key: (rew / divide_rewards) for key, rew in reward.items()}

            # Saving to memory
            self.memory.store(stacked_obs, action, reward, logprob, done)

            # Frame stacking
            for agent_id, agent in self.agents.items():
                agent.storage["last_obs"] = obs[agent_id]

            # Handle episode/loop ending
            if finish_episode and step + 1 == num_steps:
                end_flag = True

            # Update the current obs - either reset, or keep going
            if all(done.values()):  # episode is over
                if include_last:  # record the last observation along with placeholder action/reward/logprob
                    self.memory.store(next_obs, action, reward, logprob, done)
                obs = self.env.reset()
                if self.tuple_mode:
                    obs = convert_obs_to_dict(obs, self.agent_ids)

                obs = {key: preprocess_frame(obs_) for key, obs_ in obs.items()}

                # Frame stacking
                for agent_id, agent in self.agents.items():
                    agent.storage["last_obs"] = obs[agent_id]

                # Episode mode handling
                episode += 1
                if episode == num_episodes:
                    break
                # Step mode with episode finish handling
                if end_flag:
                    break
            else:  # keep going
                obs = next_obs

        return self.memory.get_torch_data()

    def reset(self):
        self.memory.reset()


class VecCollector:
    """
    Class to perform data collection from two agents.
    """

    def __init__(self, agents: Dict[str, Agent], env: VectorizedEnvSP, tuple_mode: bool = True):
        self.agents = agents
        self.agent_ids: List[str] = list(self.agents.keys())
        self.env = env
        self.memory = Memory(self.agent_ids)

        self.tuple_mode = tuple_mode

    def collect_data(self,
                     deterministic: Optional[Dict[str, bool]] = None,
                     disable_tqdm: bool = True,
                     max_steps: int = 600,
                     reset_memory: bool = True,
                     include_last: bool = False,
                     divide_rewards: Optional[int] = 10,
                     use_gpu: bool = False) -> DataBatch:

        if deterministic is None:
            deterministic = {agent_id: False for agent_id in self.agent_ids}

        if reset_memory:
            self.reset()

        # obs: Union[Tuple, Dict]
        obs = self.env.reset()

        if self.tuple_mode:  # Convert obs to dict
            obs = convert_obs_to_dict(obs, self.agent_ids)

        obs = {key: preprocess_frame(obs_) for key, obs_ in obs.items()}

        for agent_id, agent in self.agents.items():
            agent.storage["last_obs"] = obs[agent_id]

        for step in trange(max_steps, disable=disable_tqdm):
            # Compute the action for each agent

            stacked_obs = {}
            for agent_id, agent in self.agents.items():
                stacked_obs[agent_id] = np.stack([obs[agent_id], agent.storage.get("last_obs")], axis=1)

            with torch.no_grad():
                if use_gpu:
                    action_info = {  # action, logprob
                        agent_id: self.agents[agent_id].compute_actions(torch.tensor(stacked_obs[agent_id]).cuda(),
                                                                        deterministic[agent_id])
                        for agent_id in self.agent_ids
                    }
                else:
                    action_info = {  # action, logprob
                        agent_id: self.agents[agent_id].compute_actions(torch.tensor(stacked_obs[agent_id]).cpu(),
                                                                        deterministic[agent_id])
                        for agent_id in self.agent_ids
                    }

            # Unpack the actions
            action = {agent_id: action_info[agent_id][0].cpu() for agent_id in self.agent_ids}
            logprob = {agent_id: action_info[agent_id][1].cpu() for agent_id in self.agent_ids}

            # Actual step in the environment

            if self.tuple_mode:  # Convert action to env-compatible
                env_action = convert_action_to_env(action, self.agent_ids)
            else:
                env_action = action

            next_obs, reward, done, info = self.env.step(env_action)
            if self.tuple_mode:  # Convert outputs to dicts
                next_obs = convert_obs_to_dict(next_obs, self.agent_ids)
                reward = convert_obs_to_dict(reward, self.agent_ids)
                done = {agent_id: done for agent_id in self.agent_ids}

            next_obs = {key: preprocess_frame(obs_) for key, obs_ in next_obs.items()}

            if divide_rewards:
                reward = {key: (rew / divide_rewards) for key, rew in reward.items()}

            # Saving to memory
            self.memory.store(stacked_obs, action, reward, logprob, done)

            # Frame stacking
            for agent_id, agent in self.agents.items():
                agent.storage["last_obs"] = obs[agent_id]

            # Handle episode/loop ending

            # Update the current obs - either reset, or keep going
            if all(np.prod(val) for val in done.values()):  # episode is over
                if include_last:  # record the last observation along with placeholder action/reward/logprob
                    self.memory.store(next_obs, action, reward, logprob, done)

                break

            else:  # keep going
                obs = next_obs

        data_batch = self.memory

        final_batch = {}
        for agent_id in self.agent_ids:
            obs = torch.tensor(data_batch['observations']['Agent0']).transpose(0, 1)
            actions = torch.stack(data_batch['actions']['Agent0']).T
            rewards = torch.tensor(data_batch['rewards']['Agent0']).T
            logprobs = torch.stack(data_batch['logprobs']['Agent0']).T
            dones = torch.tensor(data_batch['dones']['Agent0']).T

            valid_obs = []
            valid_act = []
            valid_rew = []
            valid_log = []
            valid_don = []

            for obs_, action_, reward_, logprob_, done_ in zip(obs, actions, rewards, logprobs, dones):
                mask = ~torch.isnan(reward_)
                valid_obs.append(obs_[mask])
                valid_act.append(action_[mask])
                valid_rew.append(reward_[mask])
                valid_log.append(logprob_[mask])
                valid_don.append(done_[mask])

            valid_obs = {agent_id: torch.cat(valid_obs)}
            valid_act = {agent_id: torch.cat(valid_act)}
            valid_rew = {agent_id: torch.cat(valid_rew)}
            valid_log = {agent_id: torch.cat(valid_log)}
            valid_don = {agent_id: torch.cat(valid_don)}

        return {
            "observations": valid_obs,
            "actions": valid_act,
            "rewards": valid_rew,
            "logprobs": valid_log,
            "dones": valid_don,
        }

    def reset(self):
        self.memory.reset()


if __name__ == '__main__':
    from envs import foraging_env_creator
    from models import MLPModel, LSTMModel

    # env = foraging_env_creator({})
    #
    # agent_ids = ["Agent0", "Agent1"]
    #
    # agents: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents, env)
    #
    # data_steps = runner.collect_data(num_steps=1000, disable_tqdm=False)
    # data_episodes = runner.collect_data(num_episodes=2, disable_tqdm=False)
    # print(data_episodes['observations']['Agent0'].shape)
    # generate_video(data_episodes['observations']['Agent0'], 'vids/video.mp4')
