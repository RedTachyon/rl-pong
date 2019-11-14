import numpy as np

import gym

from rollout import Collector
from models import MLPModel, LSTMModel, RelationModel
from agents import Agent
from utils import discount_rewards_to_go
from visualize import generate_video
from trainers import PPOTrainer

import torch
from torch import optim
import torch.nn.functional as F

from typing import Dict

from tqdm import trange

import time

SUBGOALS = 4

env_config = {
    "rows": 7,
    "cols": 7,
    "subgoals": SUBGOALS,
    "random_positions": True,
    "max_steps": 100,
    #     "seed": 8
}
env = foraging_env_creator(env_config)

agent_config = {
    # SHARED
    "input_size": (3 + SUBGOALS) * 3,
    "num_actions": 5,
    "activation": F.leaky_relu,

    # MLP
    "hidden_sizes": (64, 64),

    # LSTM
    "pre_lstm_sizes": (32, ),
    "lstm_nodes": 32,
    "post_lstm_sizes": (32, ),

    # Rel
    "num_subgoals": SUBGOALS,
    "emb_size": 16,
    "rel_hiddens": (64, 64, 64, 64, 64, 64, 64, ),
    "mlp_hiddens": (32, ),
}

agent_ids = ["Agent0", "Agent1"]
agents: Dict[str, Agent] = {
    agent_id: Agent(MLPModel(agent_config), name=agent_id)
    for agent_id in agent_ids
}

trainer_config = {
    "tensorboard_name": "ppo_foraging",
    "batch_size": 10000,
    "value_loss_coeff": 1.,
    "ppo_steps": 100,
}

trainer = PPOTrainer(agents, env, config=trainer_config)

trainer.train(100)
