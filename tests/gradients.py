import numpy as np

from rollout import Collector
from models import MLPModel, LSTMModel
from agents import Agent
from utils import discount_rewards_to_go
from envs import foraging_env_creator
from visualize import generate_video

import torch
from torch import optim

from typing import Dict
import unittest


class TestGradientPropagation(unittest.TestCase):
    def test_bptt(self):
        """
        With BPTT turned on, generate two episodes. If gradients are computed from a step in the second episode,
        the gradient should appear in that step (#1), flow to previous steps in that episode (#2),
        but not to the future (#3) or to the previous episode (#4)
        """
        env = foraging_env_creator({})

        agent_ids = ["Agent0", "Agent1"]
        agents: Dict[str, Agent] = {
            agent_id: Agent(LSTMModel({}), name=agent_id)
            for agent_id in agent_ids
        }

        runner = Collector(agents, env)
        data_batch = runner.collect_data(num_episodes=2, use_tqdm=False, break_gradients=False)
        obs_batch = data_batch['observations']['Agent0']
        action_batch = data_batch['actions']['Agent0']
        state_batch = data_batch['states']['Agent0']
        reward_batch = data_batch['rewards']['Agent0']
        done_batch = data_batch['dones']['Agent0']

        logprob_batch, value_batch, entropy_batch = agents['Agent0'].evaluate_actions(obs_batch,
                                                                                      action_batch,
                                                                                      state_batch)

        for state in state_batch:
            state[0].retain_grad()
            state[1].retain_grad()

        mask = torch.zeros_like(logprob_batch)
        mask[-5] = 1

        (logprob_batch * mask).sum().backward()

        base_grad_sum = np.sum(np.abs(state_batch[-5][0].grad.numpy()))
        future_grad_sum = np.sum(np.abs(state_batch[-4][0].grad.numpy()))
        previous_grad_sum = np.sum(np.abs(state_batch[-6][0].grad.numpy()))
        prev_ep_grad_sum = np.sum(np.abs(state_batch[5][0].grad.numpy()))

        self.assertNotEqual(base_grad_sum, 0.)      # 1
        self.assertNotEqual(previous_grad_sum, 0.)  # 2
        self.assertEqual(future_grad_sum, 0.)       # 3
        self.assertEqual(prev_ep_grad_sum, 0.)      # 4

    def test_bp(self):
        """
        With BPTT turned off, the gradient should appear in the step where it's been computed (#1),
        but nowhere else (#2-4)
        """
        env = foraging_env_creator({})

        agent_ids = ["Agent0", "Agent1"]
        agents: Dict[str, Agent] = {
            agent_id: Agent(LSTMModel({}), name=agent_id)
            for agent_id in agent_ids
        }

        runner = Collector(agents, env)
        data_batch = runner.collect_data(num_episodes=2, use_tqdm=False, break_gradients=True)
        obs_batch = data_batch['observations']['Agent0']
        action_batch = data_batch['actions']['Agent0']
        state_batch = data_batch['states']['Agent0']
        reward_batch = data_batch['rewards']['Agent0']
        done_batch = data_batch['dones']['Agent0']

        logprob_batch, value_batch, entropy_batch = agents['Agent0'].evaluate_actions(obs_batch,
                                                                                      action_batch,
                                                                                      state_batch)

        for state in state_batch:
            state.retain_grad()

        mask = torch.zeros_like(logprob_batch)
        mask[-5] = 1

        (logprob_batch * mask).sum().backward()

        base_grad_sum = np.sum(np.abs(state_batch[0].grad[-5].numpy()))
        future_grad_sum = np.sum(np.abs(state_batch[0].grad[-4].numpy()))
        previous_grad_sum = np.sum(np.abs(state_batch[0].grad[-6].numpy()))
        prev_ep_grad_sum = np.sum(np.abs(state_batch[0].grad[5].numpy()))

        self.assertNotEqual(base_grad_sum, 0.)   # 1
        self.assertEqual(future_grad_sum, 0.)    # 2
        self.assertEqual(previous_grad_sum, 0.)  # 3
        self.assertEqual(prev_ep_grad_sum, 0.)   # 4


if __name__ == '__main__':
    unittest.main()