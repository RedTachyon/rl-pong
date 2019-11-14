# Pong AI via multi-agent training

## Quickstart for understanding the code

### MultiAgentEnv

This is the general structure template for a multi-agent environment. Everything is handled
with dictionaries. For example, with agents named ["Agent0", "Agent1"], an action where 
each of them takes the action `0` is ```{"Agent0": 0, "Agent1": 0}```. Observations, rewards etc are
treated analogously

### BaseModel (`models.py`)

The basic abstraction used in this code. It's just a PyTorch model equipped with certain utilities 
for handling recurrent policies (via the hidden `state`). Subclasses should overwrite the 
`.forward()` method (standard PyTorch module method) and `.get_initial_state()` - in case of 
nonrecurrent policies, just return an empty tuple. For recurrent policies, return a tuple with the state vectors
of the appropriate size (e.g. for LSTMs it's h and c)

### Agent (`agents.py`)

A wrapper around the BaseModel that handles its interactions with the environment, and is able to 
compute some things that are necessary for PPO. Should work with any model, but evaluate_actions might
need some more work for recurrent policies for speed. Also, might need some attention for convolutional models,
but might also not if the model is appropriately implemented.

### Collector (`rollout.py`)

This class is responsible for gathering data and nothing else. It holds an environment and
a number of agents that will act in that environment. It also holds a `Memory` object that is basically
a fancy dictionary for storing and basic processing of the data. In `Memory`, note the `.get_torch_data()`
method - it gathers all the stored experience and converts it into a dictionary of tensors, ready for usage

### Trainer (`trainers.py`)

This class performs the actual training. Currently only PPO is included so the 
general format is not specified, but in this case, the main methods are `.train_on_data()` which
performs a single PPO update using the passed batch of data, and `.train()` which executes a full training
loop, including data collection using a `Collector` and a PPO update with each batch.

Could at some point be refactored to separate out the training logic from the PPO logic.


### General remarks

All hyperparameters etc. are handled by configs. These should all be easily serializable so that 
they can be saved (via pickle) along with the TensorBoard logs

### Tensorboard

The training is automatically logged into TensorBoard as long as `tensorboard_name` is passed in the config.
To view the training graphs, run `tensorboard --logdir ~/tb_logs --bind_all` and see them at `localhost:6006`
in your browser. (alternatively, skip --bind_all and go to the url displayed in the terminal)