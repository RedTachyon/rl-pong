from typing import Dict, Tuple

from torch import nn as nn, Tensor
from torch.distributions import Distribution, Categorical
from torch.nn import functional as F

from rollout import Evaluator
from trainers import PPOTrainer
from agents import Agent
from models import MLPModel, LSTMModel
from envs import foraging_env_creator


env_config = {
    "rows": 7,
    "cols": 7,
    "subgoals": 2,
    "random_positions": True,
    "max_steps": 100
}

env = foraging_env_creator(env_config)

agent_ids = ["Agent0", "Agent1"]
agents: Dict[str, Agent] = {
    agent_id: Agent(LSTMModel({}), name=agent_id)
    for agent_id in agent_ids
}

collector = Evaluator(agents, env)
trainer = PPOTrainer(agents, config={})

data_batch = collector.rollout_steps(num_steps=1000)
old_weight = list(agents['Agent0'].model.parameters())[0].detach().numpy()

trainer.train_on_data(data_batch)

new_weight = list(agents['Agent0'].model.parameters())[0].detach().numpy()