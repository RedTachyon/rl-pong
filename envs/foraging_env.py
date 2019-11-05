import warnings
warnings.filterwarnings('ignore')

import numpy as np

from gym.spaces import Discrete, Box, Space

from pycolab.rendering import Observation
from pycolab.engine import Engine

from envs.foraging_game import create_game
from envs.constants import EMPTY, AGENT0, AGENT1, SUBGOAL, GOAL, NORTH, SOUTH, WEST, EAST, STAY, COLOUR_FG
from envs.base_env import MultiAgentEnv, StateDict, ActionDict, RewardDict, DoneDict, InfoDict

from typing import Dict, Tuple, Any, NamedTuple, List, Optional

# StateDict = Dict[str, np.ndarray]
# ActionDict = Dict[str, int]
# RewardDict = Dict[str, float]
# DoneDict = Dict[str, bool]
# InfoDict = Dict[str, Any]


class ForagingEnv(MultiAgentEnv):
    def __init__(self,
                 rows: int = 7,
                 cols: int = 7,
                 subgoals: int = 2,
                 random_positions: bool = True,
                 max_steps: int = 100,
                 **game_kwargs):

        self.action_space: Space = Discrete(5)  # NORTH, SOUTH, WEST, EAST, STAY

        # 2 agents, subgoals, goal, each (x, y, flag)
        self.observation_space: Space = Box(low=0., high=1., shape=((subgoals+3)*3,))

        self.subgoals = subgoals  # subgoal count
        self.random_positions = random_positions

        self.game: Engine = Engine(rows=rows, cols=cols)  # Placeholder
        self.agent0_name: Optional[str] = None
        self.agent1_name: Optional[str] = None
        self.agents: List[str] = []

        self._state: Observation = Observation(None, None)  # Holds the entire game state
        self._current_obs: StateDict = self._agent_dict(None)  # Holds the dense state representation

        self._subgoal_positions: np.ndarray = np.array([])

        self.rows = rows
        self.cols = cols

        self.game_settings = game_kwargs

        self.max_steps = max_steps
        self.current_step = 0
        self.reset()

    def reset(self) -> StateDict:
        """
        Restarts the environment using settings passed during initialization.
        :return: StateDict, a state vector for each agent
        """
        self.game: Engine = create_game(rows=self.rows,
                                        cols=self.cols,
                                        subgoals=self.subgoals,
                                        random_positions=self.random_positions,
                                        **self.game_settings)

        self.agent0_name = self.game.things[AGENT0].name
        self.agent1_name = self.game.things[AGENT1].name

        self.agents = [self.agent0_name, self.agent1_name]

        obs, reward, discount = self.game.its_showtime()  # Start the game
        self._subgoal_positions = np.argwhere(obs.layers[SUBGOAL])

        self._state: Observation = obs
        self._current_obs: StateDict = self._get_obs(obs)

        self.current_step = 0

        # print(self._subgoal_positions)
        # print(np.argwhere(obs.layers[SUBGOAL]))
        # print(obs.layers[SUBGOAL])

        return self._current_obs

    def step(self, action_dict: ActionDict) -> Tuple[StateDict, RewardDict, DoneDict, InfoDict]:
        """
        Performs a step in the game.

        :param action_dict: ActionDict, agent_name-indexed dictionary of integers
        :return: state, reward, done, info: standard format in MAEnv, similar to gym except they're dicts
        """
        # Perform a step in the game engine

        obs, reward, discount = self.game.play(action_dict)

        # Update the stored state
        self._state: Observation = obs
        self._current_obs: StateDict = self._get_obs(obs)

        # Build the feedback dictionaries
        state: StateDict = self._current_obs
        reward: RewardDict = self._agent_dict(reward)  # Shared reward
        done: DoneDict = self._agent_dict(self.game.game_over)
        done["__all__"] = self.game.game_over
        info: InfoDict = {}

        self.current_step += 1

        if self.current_step > self.max_steps:
            self.game.the_plot.terminate_episode()
            done: DoneDict = self._agent_dict(True)
            done["__all__"] = True

        return state, reward, done, info

    def _step(self, action_list: List[int]) -> Tuple[StateDict, RewardDict, DoneDict, InfoDict]:
        """
        Usage discouraged in actual code. Helpful shortcut to test the environment interactively
        """
        action_dict = {self.agent0_name: action_list[0], self.agent1_name: action_list[1]}
        return self.step(action_dict)

    def draw(self):
        """
        Draws a simple representation of the environment in the terminal
        """
        for row in self.state.board:
            print(row.tostring().decode('ascii'))

    def _get_obs(self, obs: Observation) -> StateDict:
        """
        Builds the state in a format similar to the old implementation:
        [own_coords, agent_coords, [subgoal_coords], goal_coords]
        However, each coords is its own 2-array, so the result is a 2D array (currently at least)
        :return:
        """
        board, layers = obs
        shape_array = np.array([self.rows, self.cols])

        agent0_coords = (np.argwhere(layers[AGENT0]) / shape_array)  # .ravel()  # assume there's only one agent0
        agent0_coords = np.append(agent0_coords, [[0]], axis=1)  # Compatibility with the subgoal alive flags

        agent1_coords = (np.argwhere(layers[AGENT1]) / shape_array)  # .ravel()  # same as above
        agent1_coords = np.append(agent1_coords, [[0]], axis=1)

        # Add the 1/0 flags to alive/dead subgoals
        subgoal_coords = []
        for (row_, col_) in self._subgoal_positions:
            alive = 1 if layers[SUBGOAL][row_, col_] else 0
            subgoal_coords.append((row_, col_, alive))

        subgoal_coords = np.array(subgoal_coords) / [self.rows, self.cols, 1]

        goal_coords = np.argwhere(layers[GOAL]) / shape_array
        goal_coords = np.append(goal_coords, [[0]], axis=1)

        dense_obs: StateDict = {
            self.agent0_name:
                np.concatenate((agent0_coords, agent1_coords, subgoal_coords, goal_coords), axis=0).ravel()
                                                                                                   .astype(np.float32),

            self.agent1_name:
                np.concatenate((agent1_coords, agent0_coords, subgoal_coords, goal_coords), axis=0).ravel()
                                                                                                   .astype(np.float32)
        }

        return dense_obs

    def _agent_dict(self, value: Any):
        """
        Helper function to create e.g. a RewardDict with an identical value for both agents.
        """
        return {self.agent0_name: value, self.agent1_name: value}

    def render(self):
        self.draw()
        print()

    def seed(self, value):
        pass

    @property
    def state(self):
        return self._state

    @property
    def current_obs(self):
        return self._current_obs


def foraging_env_creator(config: Dict) -> MultiAgentEnv:
    return ForagingEnv(**config)
