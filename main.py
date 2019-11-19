import numpy as np

import gym

from rollout import Collector
from models import MLPModel, LSTMModel, RelationModel
from agents import Agent
from utils import discount_rewards_to_go
from visualize import generate_video
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
    "input_size": 6,
    "num_actions": 3,
    "stack_size": 3,
    "activation": "leaky_relu",

    # MLP
    "hidden_sizes": (64, 64),

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
    "tensorboard_name": "pong_test",
    "batch_size": 1000,
    "value_loss_coeff": 1.,
    "ppo_steps": 50,
    "tuple_mode": True,
    "stack_size": 3
}

trainer = PPOTrainer(agents, env, config=trainer_config)

trainer.train(100)
