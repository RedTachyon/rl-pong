import numpy as np

from pycolab import human_ui
from pycolab.things import Backdrop, Drape, Sprite
from pycolab.prefab_parts.sprites import MazeWalker
from pycolab.engine import Engine
from pycolab.plot import Plot

import curses

# from ray.rllib.env import MultiAgentEnv

from typing import List, Tuple, Dict, NamedTuple, Union

from collections import defaultdict

from envs.constants import EMPTY, AGENT0, AGENT1, SUBGOAL, GOAL, NORTH, SOUTH, WEST, EAST, STAY, COLOUR_FG, \
    SUBGOAL_REWARD, GOAL_REWARD, STEP_REWARD


class Forager(MazeWalker):
    """
    The agent representation in the Pycolab game.
    Can walk anywhere within the board, collect subgoals and reach the goal
    """

    def __init__(self,
                 corner: NamedTuple,
                 position: NamedTuple,
                 character: str,
                 name: str = "Agent_0"):

        super().__init__(corner, position, character, impassable='', confined_to_board=True)
        self.name = name

    def update(self,
               actions: Dict[str, int],
               board: np.ndarray,
               layers: Dict[str, np.ndarray],
               backdrop: Backdrop,
               things: Dict[str, Union[Sprite, Drape]],
               the_plot: Plot):

        if actions is None:
            return

        try:  # Check if agent is present in the action dict - if not, take no action (aka stay)
            action = actions[self.name]
        except KeyError:
            the_plot.log(f"Name of the agent {self.name} not present in the action dict")
            self._stay(board, the_plot)
            return

        # Take the action, using methods from the MazeWalker template
        if action == NORTH:
            self._north(board, the_plot)
        elif action == SOUTH:
            self._south(board, the_plot)
        elif action == WEST:
            self._west(board, the_plot)
        elif action == EAST:
            self._east(board, the_plot)
        elif action == STAY:
            self._stay(board, the_plot)
        else:
            the_plot.log(f"Invalid action {action} detected for agent {self.name}")


class Subgoal(Drape):
    """
    Drape representing the subgoals. They're all stored inside an instance of this object,
    with coordinates stored in the positions argument.
    """

    def __init__(self, curtain: np.ndarray,
                 character: str,
                 positions: List[Tuple[int, int]]):

        super().__init__(curtain, character)

        # Set the initial positions of the subgoals to True
        for (s_row, s_col) in positions:
            self.curtain[s_row, s_col] = True

    def update(self,
               actions: Dict[str, int],
               board: np.ndarray,
               layers: Dict[str, np.ndarray],
               backdrop: Backdrop,
               things: Dict[str, Union[Sprite, Drape]],
               the_plot: Plot):

        agent_positions = (things[AGENT0].position, things[AGENT1].position)

        if self.curtain[agent_positions[0]]:  # Agent0 collecting subgoals
            the_plot.log(f"Food collected by {things[AGENT0].name}")
            the_plot.add_reward(SUBGOAL_REWARD)
            self.curtain[agent_positions[0]] = False  # Remove the subgoal
            # self.positions.remove(agent_positions[0])

        if self.curtain[agent_positions[1]] and agent_positions[0] != agent_positions[1]:  # Agent1 collecting subgoals
            the_plot.log(f"Food collected by {things[AGENT1].name}")
            the_plot.add_reward(SUBGOAL_REWARD)
            self.curtain[agent_positions[1]] = False  # Remove the subgoal
            # self.positions.remove(agent_positions[1])


class Goal(Sprite):
    """
    Stationary goal, responsible for giving negative rewards with time, and finishing the game.
    """

    def __init__(self, corner, position, character):
        super().__init__(corner, position, character)

    def update(self,
               actions: Dict[str, int],
               board: np.ndarray,
               layers: Dict[str, np.ndarray],
               backdrop: Backdrop,
               things: Dict[str, Union[Sprite, Drape]],
               the_plot: Plot):

        the_plot.add_reward(STEP_REWARD)  # Small negative reward at each time step

        subgoals = things[SUBGOAL].curtain
        if not subgoals.any():  # If there's no more subgoals
            agent_positions = (things[AGENT0].position, things[AGENT1].position)
            if self.position == agent_positions[0]:  # if the first agent collects the goal
                the_plot.log(f"Goal reached by {things[AGENT0].name}")
                the_plot.add_reward(GOAL_REWARD)
                the_plot.terminate_episode()

            elif self.position == agent_positions[1]:  # if the second agent collects the goal
                the_plot.log(f"Goal reached by {things[AGENT1].name}")
                the_plot.add_reward(GOAL_REWARD)
                the_plot.terminate_episode()


class Field(Backdrop):
    """
    Backdrop for the game. Doesn't really do anything, but is required by pycolab
    """

    def __init__(self, curtain, palette):
        super().__init__(curtain, palette)

        # Fill the backdrop with a constant value.
        start = np.full_like(curtain, palette[EMPTY], dtype=np.uint8)
        np.copyto(self.curtain, start)


def create_game(rows: int = 7, cols: int = 7, subgoals: int = 4,
                random_positions: bool = False,
                *,
                seed: int = None,
                agent_positions: List[Tuple[int, int]] = None,
                subgoal_positions: List[Tuple[int, int]] = None,
                goal_position: Tuple[int, int] = None) -> Engine:
    """
    Sets up the pycolab foraging game.

    :return: Engine object
    """
    if random_positions:
        if seed is not None:
            np.random.seed(seed)
        item_count = subgoals + 3  # subgoals, 2 agents, 1 goal
        row_idx = np.arange(rows)
        col_idx = np.arange(cols)

        grid = np.stack(np.meshgrid(row_idx, col_idx))
        coords = np.transpose(grid, [1, 2, 0])  # brings it to shape [rows, cols, 2]
        coords = coords.reshape(rows * cols, 2)  # list of (x, y) pairs of all possible coordinates)

        # Randomly choose rows*cols of (x, y) pairs, without replacement, as starting positions
        random_coords = coords[np.random.choice(rows * cols, size=item_count, replace=False)]

        agent_positions = random_coords[0:2]
        subgoal_positions = random_coords[2:-1]
        goal_position = random_coords[-1]
    else:
        # Example fixed starting positions, only for 4 subgoals on 7x7 grid
        if agent_positions is None:
            agent_positions = [(1, 1), (1, 5)]
        if subgoal_positions is None:
            subgoal_positions = [(3, 3), (4, 4), (6, 2), (0, 3)]
        if goal_position is None:
            goal_position = (6, 6)

    engine = Engine(rows=rows, cols=cols, occlusion_in_layers=False)
    engine.set_backdrop(EMPTY, Field)

    engine.update_group('1. Agents')
    engine.add_sprite(AGENT0, agent_positions[0], Forager, name="Agent0")
    engine.add_sprite(AGENT1, agent_positions[1], Forager, name="Agent1")

    engine.update_group('2. Subgoals')
    engine.add_drape(SUBGOAL, Subgoal, positions=subgoal_positions)

    engine.update_group('3. Goal')
    engine.add_sprite(GOAL, goal_position, Goal)

    return engine


if __name__ == '__main__':
    game = create_game(rows=20, cols=50, subgoals=50, random_positions=True)

    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: {"Agent0": NORTH, "Agent1": STAY},
                         curses.KEY_DOWN: {"Agent0": SOUTH, "Agent1": STAY},
                         curses.KEY_LEFT: {"Agent0": WEST, "Agent1": STAY},
                         curses.KEY_RIGHT: {"Agent0": EAST, "Agent1": STAY},

                         'w': {"Agent0": STAY, "Agent1": NORTH},
                         's': {"Agent0": STAY, "Agent1": SOUTH},
                         'a': {"Agent0": STAY, "Agent1": WEST},
                         'd': {"Agent0": STAY, "Agent1": EAST},

                         'f': {"Agent0": STAY, "Agent1": STAY}
                         },
        delay=1000,
        colour_fg=COLOUR_FG,
    )

    ui.play(game)
