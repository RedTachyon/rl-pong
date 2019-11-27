import numpy as np

import gym

from rollout import Collector
from models import MLPModel, LSTMModel, RelationModel
from agents import Agent
from utils import discount_rewards_to_go
from trainers import PPOTrainer

from envs import wimblepong

import torch
from torch import optim
import torch.nn.functional as F

from typing import Dict

from tqdm import trange

import time

env = gym.make('WimblepongSimpleAI-v0')

agent_config = {
    # SHARED
    "input_size": 12,  # 2-stacked obs
    "num_actions": 3,
    "activation": "relu",

    # MLP
    "hidden_sizes": (64, ) * 7,
}

agent_ids = ["Agent0"]#, "Agent1"]
agents: Dict[str, Agent] = {
    agent_id: Agent(MLPModel(agent_config), name=agent_id)
    for agent_id in agent_ids
}


trainer_config = {
        # Trainer settings
        "agents_to_optimize": None,  # ids of agents that should be optimized
        "batch_size": 2000,
        # Agent settings
        "optimizer": "adam",
        "optimizer_kwargs": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-7,
            "weight_decay": 0,
            "amsgrad": False
        },
        "gamma": 0.95,  # Discount factor

        # PPO settings
        "ppo_steps": 25,
        "eps": 0.1,  # PPO clip parameter
        "target_kl": 0.01,  # KL divergence limit
        "value_loss_coeff": 0.01,
        "entropy_coeff": 0.001,

        # Tensorboard settings
        "tensorboard_name": "relu",

        # Compatibility
        "tuple_mode": True,

        }

trainer = PPOTrainer(agents, env, config=trainer_config)

trainer.train(10000, finish_episode=True, divide_rewards=10)
