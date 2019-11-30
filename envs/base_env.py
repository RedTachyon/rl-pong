import numpy as np
import gym
from typing import Dict, Any, Tuple, Callable, List

StateDict = Dict[str, np.ndarray]
ActionDict = Dict[str, Any]
RewardDict = Dict[str, float]
DoneDict = Dict[str, bool]
InfoDict = Dict[str, Any]


class MultiAgentEnv(gym.Env):
    """
    Base class for a gym-like environment for multiple agents. An agent is identified with its id (string),
    and most interactions are communicated through that API (actions, states, etc)
    """
    def __init__(self):
        self.config = {}
        raise NotImplementedError

    def reset(self) -> StateDict:
        """
        Resets the environment and returns the state.
        Returns:
            A dictionary holding the state visible to each agent.
        """
        raise NotImplementedError

    def step(self, action_dict: ActionDict) -> Tuple[StateDict, RewardDict, DoneDict, InfoDict]:
        """
        Executes the chosen actions for each agent and returns information about the new state.

        Args:
            action_dict: dictionary holding each agent's action

        Returns:
            states: new state for each agent
            rewards: reward obtained by each agent
            dones: whether the environment is done for each agent
            infos: any additional information
        """
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


class VectorizedEnvSP(gym.Env):
    def __init__(self, env_creator: Callable[[], gym.Env], num_envs: int):
        self.num_envs = num_envs
        self.envs = [
            env_creator() for _ in range(self.num_envs)
        ]

        self.finishes = [False for _ in self.envs]

        sample_env = env_creator()
        self.zero_obs = np.zeros_like(sample_env.reset())

    def reset(self) -> np.ndarray:
        obs_batch = [env.reset() for env in self.envs]
        self.finishes = [False for _ in self.envs]
        return np.stack(obs_batch)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        assert len(actions) == len(self.envs), AssertionError("Need to pass all the actions")
        obs_batch = []
        reward_batch = []
        done_batch = []
        info_batch = []
        for i, (env, action, finished) in enumerate(zip(self.envs, actions, self.finishes)):
            if not finished:
                obs, reward, done, info = env.step(action)
            else:
                obs = self.zero_obs
                reward = np.nan
                done = True
                info = {}

            obs_batch.append(obs)
            reward_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)

            if done:
                self.finishes[i] = True

        obs_batch = np.array(obs_batch)
        reward_batch = np.array(reward_batch)
        done_batch = np.array(done_batch)

        return obs_batch, reward_batch, done_batch, info_batch
