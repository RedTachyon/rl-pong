from typing import Dict, Tuple, Callable, Iterable

import numpy as np

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical
from torch.nn import functional as F

from layers import RelationLayer
from utils import with_default_config, get_activation


class BaseModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        default_config = {
            "img_value_range": 255
        }

        self._stateful = False
        self.config = config
        self.value_range = default_config.get("img_value_range")

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
            "stack_size": 2,
            "num_actions": 5,
            "hidden_sizes": (64, 64),
            "activation": "leaky_relu",
            "img_value_range": 255
        }
        self.config = with_default_config(config, default_config)

        input_size: int = self.config.get("input_size")
        stack_size: int = self.config.get("stack_size")
        num_actions: int = self.config.get("num_actions")
        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))

        layer_sizes = (input_size * stack_size, ) + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(layer_sizes[-1], num_actions)
        self.value_head = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x: Tensor, state: Tuple = ()) -> Tuple[Distribution, Tensor, Tuple[Tensor, Tensor]]:
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


class RelationModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "num_actions": 5,
            "num_subgoals": 2,
            "emb_size": 4,
            "rel_hiddens": (16, 16, ),
            "mlp_hiddens": (16, ),
            "activation": "leaky_relu"
        }
        self.config = with_default_config(config, default_config)

        self.relation_layer = RelationLayer(self.config)

        self.policy_head = nn.Linear(self.config["mlp_hiddens"][-1], self.config["num_actions"])
        self.value_head = nn.Linear(self.config["mlp_hiddens"][-1], 1)

    def forward(self, x: Tensor, state: Tuple) -> Tuple[Distribution, Tensor, Tuple[Tensor, Tensor]]:
        x = self.relation_layer(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value, state

    def get_initial_state(self) -> Tuple:
        return ()


class LSTMModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        self._stateful = True

        default_config = {
            "input_size": 15,
            "num_actions": 5,
            "pre_lstm_sizes": (32, ),
            "lstm_nodes": 32,
            "post_lstm_sizes": (32, ),
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
        post_layers = (lstm_nodes, ) + post_lstm_sizes

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



class CNNMLPModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_size": (210, 160),
            "stack_size": 3,
            "num_actions": 5,
            "hidden_sizes": (64, 64),
            "activation": "leaky_relu",
        }
        self.config = with_default_config(config, default_config)

        input_size: int = self.config.get("input_size")
        stack_size: int = self.config.get("stack_size")
        num_actions: int = self.config.get("num_actions")
        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=stack_size, out_channels=32, kernel_size=8, stride=4),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        ])

        conv_out_size = self._get_conv_out([stack_size, *input_size])
        layer_sizes = (conv_out_size, ) + hidden_sizes


        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(layer_sizes[-1], num_actions)
        self.value_head = nn.Linear(layer_sizes[-1], 1)


    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        for layer in self.conv_layers:
            x = layer(x)
            x = self.activation(x)

        return int(np.prod(x.size()))

    def forward(self, x: Tensor, state: Tuple = ()) -> Tuple[Distribution, Tensor, Tuple[Tensor, Tensor]]:
        # noinspection PyTypeChecker

        # normalize and cast to float
        x = x.float() / self.value_range

        for layer in self.conv_layers:
            x = layer(x)
            x = self.activation(x)

        x = x.flatten(1, -1)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value, state

    def get_initial_state(self):
        return ()

if __name__ == '__main__':
    policy = LSTMModel({})
    data = torch.randn(2, 15)

    action_dist, value_, state_ = policy(data, ())
