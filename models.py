from typing import Dict, Tuple, Callable, Iterable

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical
from torch.nn import functional as F

from utils import with_default_config


class BaseModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self._stateful = False

    def forward(self, x: Tensor, state: Tuple) -> Tuple[Distribution, Tensor, Tuple[Tensor, Tensor]]:
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
            "activation": F.relu,
        }
        config = with_default_config(config, default_config)

        input_size: int = config.get("input_size")
        num_actions: int = config.get("num_actions")
        hidden_sizes: Tuple[int] = config.get("hidden_sizes")
        self.activation: Callable = config.get("activation")

        layer_sizes = (input_size, ) + hidden_sizes

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


class LSTMModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        self._stateful = True

        default_config = {
            "input_size": 15,
            "num_actions": 5,
            "pre_lstm_layers": (32, ),
            "lstm_nodes": 32,
            "post_lstm_layers": (32, ),
            "activation": F.relu
        }
        config = with_default_config(config, default_config)

        # Unpack the config
        input_size: int = config.get("input_size")
        num_actions: int = config.get("num_actions")
        pre_lstm_sizes: Tuple[int] = config.get("pre_lstm_layers")
        lstm_nodes: int = config.get("lstm_nodes")
        post_lstm_sizes: Tuple[int] = config.get("post_lstm_layers")
        self.activation: Callable = config.get("activation")

        pre_layers = (input_size,) + pre_lstm_sizes
        post_layers = (lstm_nodes, ) + post_lstm_sizes

        self.preprocess_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(pre_layers, pre_layers[1:])
        ])

        self.postprocess_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(post_layers, post_layers[1:])
        ])

        self.lstm = nn.LSTMCell(input_size=pre_layers[-1],
                                hidden_size=lstm_nodes,
                                bias=True)

        self.policy_head = nn.Linear(post_layers[-1], num_actions)
        self.value_head = nn.Linear(post_layers[-1], 1)

    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Distribution, Tensor, Tuple[Tensor, Tensor]]:
        # noinspection PyTypeChecker
        for layer in self.preprocess_layers:
            x = layer(x)
            x = self.activation(x)

        (h_state, c_state) = self.lstm(x, state) if len(state) > 0 else self.lstm(x)

        x = h_state

        # noinspection PyTypeChecker
        for layer in self.postprocess_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value, (h_state, c_state)


if __name__ == '__main__':
    policy = LSTMModel({})
    data = torch.randn(2, 15)

    action_dist, value_, state_ = policy(data, ())
