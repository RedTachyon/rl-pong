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
    "activation": "leaky_relu",

    # MLP
    "hidden_sizes": (64, ) * 7,

    # LSTM
    "pre_lstm_sizes": (32, ),
    "lstm_nodes": 32,
    "post_lstm_sizes": (32, ),
}

agent_ids = ["Agent0"]#, "Agent1"]
agents: Dict[str, Agent] = {
    agent_id: Agent(MLPModel(agent_config), name=agent_id)
    for agent_id in agent_ids
}

trainer_config = {
    "tensorboard_name": "stacking",
    "batch_size": 2000,
    "value_loss_coeff": .01,
    "ppo_steps": 50,
    "tuple_mode": True,
    "target_kl": 0.02,
    "entropy_coeff": 0.001,
}

trainer = PPOTrainer(agents, env, config=trainer_config)

trainer.train(10000, finish_episode=True, divide_rewards=10)
