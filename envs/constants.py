EMPTY = ','
AGENT0 = 'A'
AGENT1 = 'B'
SUBGOAL = 'S'
GOAL = 'G'
NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3
STAY = 4
COLOUR_FG = {
    EMPTY:   (400, 400, 400),      # Gray background
    AGENT0:  (999, 0,   0),   # Red agent
    AGENT1:  (0,   0,   999),   # Blue agent
    SUBGOAL: (0,   999, 0),  # Green subgoal
    GOAL:    (999, 999, 0),   # Yellow goal
}
SUBGOAL_REWARD = 0.01
GOAL_REWARD = 0.01
STEP_REWARD = -0.01  # change to -0.005?
