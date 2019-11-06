import numpy as np

from typing import Dict, Callable, List, Tuple, Optional

from agents import Agent
from envs import MultiAgentEnv

import torch
from torch import Tensor

from utils import append_dict, DataBatch

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
        _dones: Dict[str, List[bool]] = {agent: [] for agent in self.agents}
        _states: Dict[str, List[Tuple[Tensor, Tensor]]] = {agent: [] for agent in self.agents}

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

        def stack_states(states_: List[Tuple[Tensor, Tensor]]):
            transposed_states: Tuple[List[Tensor], ...] = tuple(list(i) for i in zip(*states_))
            # ([h1, h2, ...], [c1, c2, ...]) /\

            tensor_states = tuple(torch.cat(state_type) for state_type in transposed_states)
            # (tensor(h1, h2, ...), tensor(c1, c2, ...)) /\

            return tensor_states

        states: Dict[str, List[Tuple[Tensor, Tensor]]] = self.data["states"]
        states = self.apply_to_agent(lambda agent: stack_states(states[agent]))

        return {
            "observations": observations,  # (batch_size, obs_size) float
            "actions": actions,  # (batch_size, ) int
            "rewards": rewards,  # (batch_size, ) float
            "logprobs": logprobs,  # (batch_size, ) float
            "dones": dones,  # (batch_size, ) bool
            "states": states,  # (batch_size, 2, lstm_nodes)
        }

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return self.data.__str__()


class Evaluator:
    def __init__(self, agents: Dict[str, Agent], env: MultiAgentEnv):
        self.agents = agents
        self.agent_ids: List[str] = list(self.agents.keys())
        self.env = env
        self.memory = Memory(self.agent_ids)

    def rollout_steps(self,
                      num_steps: Optional[int] = None,
                      num_episodes: Optional[int] = None,
                      deterministic: Optional[Dict[str, bool]] = None,
                      use_tqdm: bool = False,
                      max_steps: int = 102,
                      reset_memory: bool = True,
                      include_last: bool = False) -> DataBatch:
        """
        Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

        Args:
            num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
            num_episodes: number of episodes to generate
            deterministic: whether each agent should use the greedy policy; False by default
            use_tqdm: whether a live progress bar should be displayed
            max_steps: maximum number of steps that can be taken in episodic mode, recommended just above env maximum
            reset_memory: whether to reset the memory before generating data
            include_last: whether to include the last observation in episodic mode - useful for visualizations

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

            "states":
                {
                    "Agent0": (tensor([h1, h2, ...]), tensor([c1, c2, ...])),

                    "Agent1": (tensor([h1, h2, ...]), tensor([c1, c2, ...]))
                }
        }
        """
        assert not ((num_steps is None) == (num_episodes is None)), ValueError("Exactly one of num_steps, num_episodes "
                                                                               "should receive a value")

        if deterministic is None:
            deterministic = {agent_id: False for agent_id in self.agent_ids}

        if reset_memory:
            self.reset()

        obs = self.env.reset()
        state = {
            agent_id: self.agents[agent_id].get_initial_state() for agent_id in self.agent_ids
        }

        episode = 0

        steps = num_steps if num_steps else max_steps * num_episodes
        iterator = trange(steps) if use_tqdm else range(steps)
        for step in iterator:
            # Compute the action for each agent
            # with torch.no_grad():
            action_info = {  # action, logprob, state
                agent_id: self.agents[agent_id].compute_single_action(obs[agent_id],
                                                                      state[agent_id],
                                                                      deterministic[agent_id])
                for agent_id in self.agent_ids
            }

            # Unpack the actions
            action = {agent_id: action_info[agent_id][0] for agent_id in self.agent_ids}
            logprob = {agent_id: action_info[agent_id][1] for agent_id in self.agent_ids}
            next_state = {agent_id: action_info[agent_id][2] for agent_id in self.agent_ids}

            # Actual step in the environment
            next_obs, reward, done, info = self.env.step(action)

            # Saving to memory
            self.memory.store(obs, action, reward, logprob, done, state)

            # Update the current obs and state - either reset, or keep going
            if done['Agent0'] and done['Agent1']:  # both values should always be the same anyways
                if include_last:  # record the last observation along with placeholder action/reward/logprob
                    self.memory.store(next_obs, action, reward, logprob, done, next_state)
                obs = self.env.reset()
                state = {
                    agent_id: self.agents[agent_id].get_initial_state() for agent_id in self.agent_ids
                }
                episode += 1
                if episode == num_episodes:
                    break
            else:  # keep going
                obs = next_obs
                state = next_state

        return self.memory.get_torch_data()

    def reset(self):
        self.memory.reset()


if __name__ == '__main__':
    from envs import foraging_env_creator
    from models import MLPModel

    env = foraging_env_creator({})

    agent_ids = ["Agent0", "Agent1"]

    agents: Dict[str, Agent] = {
        agent_id: Agent(MLPModel({}), name=agent_id)
        for agent_id in agent_ids
    }

    runner = Evaluator(agents, env)

    data_steps = runner.rollout_steps(num_steps=1000, use_tqdm=True)
    # data_episodes = runner.rollout_steps(num_episodes=2, use_tqdm=True, include_last=True)
    # print(data_episodes['observations']['Agent0'].shape)
    # generate_video(data_episodes['observations']['Agent0'], 'vids/video.mp4')
