import numpy as np

import gym

from rollout import Collector
from models import MLPModel, CoordConvModel, BilinearCoordPooling, SpatialSoftMaxModel, StridedConvModel
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


#env = gym.make('WimblepongSimpleAI-v0')
env = gym.make('WimblepongVisualSimpleAI-v0')

agent_config = {
    # SHARED

    "input_size": 12,  # 2-stacked obs
    "num_actions": 3,
    "activation": "relu",

    # MLP
    "hidden_sizes": (64, ) * 2,
}

agent_ids = ["Agent0"]#, "Agent1"]
agents: Dict[str, Agent] = {
    agent_id: Agent(StridedConvModel(agent_config), name=agent_id)
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
    "preserve_channels": False, #Store frames with colors
    "load_weights_from_step": '_1300',

    # PPO settings
    "ppo_steps": 25,
    "eps": 0.1,  # PPO clip parameter
    "target_kl": 0.01,  # KL divergence limit
    "value_loss_coeff": 0.01,
    "entropy_coeff": 0.01,

    # Tensorboard settings
    "tensorboard_name": "strided_conv_++entropy_--reward_bonus",

    # Compatibility
    "visual": True,
    "tuple_mode": True,
    "use_gpu": True,

}

trainer = PPOTrainer(agents, env, config=trainer_config)

trainer.train(10000, finish_episode=True, divide_rewards=10)
