from typing import Dict, Callable

from agents import Agent
from envs import MultiAgentEnv

class Evaluator:
    def __init__(self, agents: Dict[str, Agent], env: MultiAgentEnv):
        self.agents = agents
        self.env = env

    def rollout(self, deterministic: Dict[str, bool]):
        obs = self.env.reset()
        obs_batch = [obs]

        done = False
        while not done:
            pass # TODO: Put some thought in bookkeeping here

    def for_agent(self, func: Callable):
        return {
            agent: func(agent) for agent in self.agents
        }