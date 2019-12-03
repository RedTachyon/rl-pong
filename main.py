import numpy as np

import gym

from rollout import Collector
from models import MLPModel, CoordConvModel, SpatialSoftMaxModel
from agents import Agent
from utils import discount_rewards_to_go, preprocess_frame
from trainers import PPOTrainer

from envs import wimblepong

import torch
from torch import optim
import torch.nn.functional as F

from typing import Dict

from tqdm import trange

import time
import matplotlib.pyplot as plt


# env = gym.make('WimblepongSimpleAI-v0')
env = gym.make('WimblepongVisualSimpleAI-v0')

agent_config = {
    # SHARED

    "input_size": 12,  # 2-stacked obs
    "num_actions": 3,
    "activation": "relu",

    # MLP
    "hidden_sizes": (64, ) * 2,

    "load_model": True,
    "load_model_from_path": '/home/llama/tb_logs/spatial_softmax_2019-12-02_21-26-53/Agent0_1300.pt',

}

agent_ids = ["Agent0"]#, "Agent1"]
agents: Dict[str, Agent] = {
    agent_id: Agent(SpatialSoftMaxModel(agent_config), name=agent_id)
    #agent_id: Agent(CoordConvModel(agent_config), name=agent_id)
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
    "gamma": 0.999,  # Discount factor
    "preserve_channels": True, #Store frames with colors

    # PPO settings
    "ppo_steps": 25,
    "eps": 0.1,  # PPO clip parameter
    "target_kl": 0.01,  # KL divergence limit
    "value_loss_coeff": 0.01,
    "entropy_coeff": 0.01,

    # Tensorboard settings
    "tensorboard_name": "spatial_softmax",

    # Compatibility
    "tuple_mode": True,
    "use_gpu": True,

}

trainer = PPOTrainer(agents, env, config=trainer_config)

trainer.train(10000, finish_episode=True, divide_rewards=10)
