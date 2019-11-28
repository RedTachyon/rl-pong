from typing import Dict, Tuple, Callable, Iterable

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical
from torch.nn import functional as F

from layers import RelationLayer
from utils import with_default_config, get_activation


class BaseModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self._stateful = False
        self.config = config

    def forward(self, x: Tensor, state: Tuple) -> Tuple[Distribution, Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def get_initial_state(self) -> Tuple:
        raise NotImplementedError

    @property
    def stateful(self):
        return self._stateful


class MLPModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_size": 15,
            "num_actions": 5,
            "hidden_sizes": (64, 64),
            "activation": "leaky_relu",
        }
        self.config = with_default_config(config, default_config)

        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))

        layer_sizes = (input_size,) + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(layer_sizes[-1], num_actions)
        self.value_head = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x: Tensor, state: Tuple = ()) -> Tuple[Distribution, Tensor, Tuple[Tensor, Tensor]]:
        x = x.view((x.shape[0], -1))
        # noinspection PyTypeChecker
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value, state

    def get_initial_state(self):
        return ()


class CoordConvModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_shape": (100, 100),
            "num_actions": 5,
            "activation": "relu",


        }
        self.config = with_default_config(config, default_config)
        self.activation = get_activation(self.config["activation"])

        input_shape: Tuple[int, int] = self.config["input_shape"]

        self.conv_layers = nn.ModuleList([nn.Conv2d(4, 32, kernel_size=3, stride=2),  # 24x24x32
                                          nn.Conv2d(32, 32, kernel_size=3, stride=2),  # 6x6x64
                                          nn.Conv2d(32, 32, kernel_size=3, stride=1)])  # 4x4x64

        _coords_i = torch.linspace(-1, 1, input_shape[0]).view(-1, 1).repeat(1, input_shape[1])
        _coords_j = torch.linspace(-1, 1, input_shape[1]).view(1, -1).repeat(input_shape[0], 1)
        self.coords = torch.stack([_coords_i, _coords_j])

        # flatten
        self.mlp_layers = nn.ModuleList([nn.Linear()])

        self.policy_head = nn.Linear(4*4*64, self.config["num_actions"])
        self.value_head = nn.Linear(4*4*64, 1)

    def forward(self, x: Tensor, state: Tuple = ()):
        batch_size = x.shape[0]
        batch_coords = torch.stack([self.coords for _ in range(batch_size)], dim=0)
        # breakpoint()
        x = torch.cat([x, batch_coords], dim=1)
        # noinspection PyTypeChecker
        for layer in self.conv_layers:
            x = layer(x)
            x = self.activation(x)

        x = x.flatten(1, -1)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value, state

    def get_initial_state(self):
        return ()


class LSTMModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        self._stateful = True

        default_config = {
            "input_size": 15,
            "num_actions": 5,
            "pre_lstm_sizes": (32,),
            "lstm_nodes": 32,
            "post_lstm_sizes": (32,),
            "activation": "leaky_relu"
        }
        self.config = with_default_config(config, default_config)

        # Unpack the config
        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        pre_lstm_sizes: Tuple[int] = self.config.get("pre_lstm_sizes")
        lstm_nodes: int = self.config.get("lstm_nodes")
        post_lstm_sizes: Tuple[int] = self.config.get("post_lstm_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))

        pre_layers = (input_size,) + pre_lstm_sizes
        post_layers = (lstm_nodes,) + post_lstm_sizes

        self.preprocess_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(pre_layers, pre_layers[1:])
        ])

        self.lstm = nn.LSTMCell(input_size=pre_layers[-1],
                                hidden_size=lstm_nodes,
                                bias=True)

        self.postprocess_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(post_layers, post_layers[1:])
        ])

        self.policy_head = nn.Linear(post_layers[-1], num_actions)
        self.value_head = nn.Linear(post_layers[-1], 1)

    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Distribution, Tensor, Tuple[Tensor, Tensor]]:
        # noinspection PyTypeChecker
        for layer in self.preprocess_layers:
            x = layer(x)
            x = self.activation(x)

        if len(state) == 0:
            state = self.get_initial_state()

        (h_state, c_state) = self.lstm(x, state)

        x = h_state

        # noinspection PyTypeChecker
        for layer in self.postprocess_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value, (h_state, c_state)

    def get_initial_state(self) -> Tuple[Tensor, Tensor]:
        return torch.zeros(1, self.config['lstm_nodes'], requires_grad=True), \
               torch.zeros(1, self.config['lstm_nodes'], requires_grad=True)


if __name__ == '__main__':
    policy = LSTMModel({})
    data = torch.randn(2, 15)

    action_dist, value_, state_ = policy(data, ())
