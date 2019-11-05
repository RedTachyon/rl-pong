import pickle
import os

import numpy as np
import matplotlib as mpl

from PIL import Image

from typing import Dict, List


# Constants for colormap
AGENT0 = 1
AGENT1 = 2
SUBGOAL = 3
DEAD_SUBGOAL = 4
GOAL = 5
COLORS = ['white', 'red', 'blue', 'green', 'purple', 'yellow']

# Other constants
ROWS = 7
COLS = 7
PX_PER_CELL = 100

VIDEO_DIMS = PX_PER_CELL*ROWS, PX_PER_CELL*COLS

cmap = mpl.colors.ListedColormap(COLORS)  # Colormap for matplotlib


def obs_to_frame(obs: Dict[str, np.ndarray],
                 rows: int = 7,
                 cols: int = 7) -> np.ndarray:
    """
    Converts an observation from the environment into a drawable frame.

    Priority of drawing (top to bottom):
    1. Agent0
    2. Agent1
    3. Subgoals/goals (never collide)
    """

    # Convert an observation to an array of color indices
    obs: np.ndarray = obs[list(obs.keys())[0]]  # Quick trick to get the first entry of the dictionary
    agent_xy = np.round(obs[:2] * [rows, cols]).astype(int)  # first agent coords
    other_xy = np.round(obs[3:5] * [rows, cols]).astype(int)  # second agent coords

    subgoal_xy = np.round(obs[6:-3].reshape(-1, 3) * [rows, cols, 1]).astype(int)  # array of [x, y, alive] of subgoals

    goal_xy = np.round(obs[-3:-1] * [rows, cols]).astype(int)

    grid = np.zeros((rows, cols), dtype=int)

    # Order reverse of the priority
    grid[goal_xy[0], goal_xy[1]] = GOAL

    for [subgoal_x, subgoal_y, alive] in subgoal_xy:
        grid[subgoal_x, subgoal_y] = SUBGOAL if alive else DEAD_SUBGOAL

    grid[other_xy[0], other_xy[1]] = AGENT1
    grid[agent_xy[0], agent_xy[1]] = AGENT0

    return grid


def generate_video(rollout: List[Dict[str, np.ndarray]],
                   out_path: str = 'vids/video.mp4'):

    frames = [obs_to_frame(frame) for frame in rollout]
    frames = [np.uint8(cmap(frame)*255) for frame in frames]  # rescale to 0-255 RGB

    # List of images - frames in the actual video
    images = [Image.fromarray(frame).resize(VIDEO_DIMS) for frame in frames]  # resize in PIL

    # Create a directory for the frames
    try:
        os.mkdir('temp')
    except FileExistsError:
        pass

    for i, image in enumerate(images):
        image.save('temp/image%02d.png' % i)

    os.system('ffmpeg -y -framerate 2 -i ./temp/image%02d.png -pix_fmt yuv420p ' + out_path)
    os.system("rm -rf ./temp")


if __name__ == '__main__':
    with open("vids/rollout.pkl", 'rb') as f:
        rollouts = pickle.load(f)

    generate_video(rollout=rollouts[0])
