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
            "activation": F.relu,
        }
        self.config = with_default_config(config, default_config)

        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")
        self.activation: Callable = self.config.get("activation")

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

    def get_initial_state(self):
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
            "activation": F.relu
        }
        self.config = with_default_config(config, default_config)

        # Unpack the config
        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        pre_lstm_sizes: Tuple[int] = self.config.get("pre_lstm_sizes")
        lstm_nodes: int = self.config.get("lstm_nodes")
        post_lstm_sizes: Tuple[int] = self.config.get("post_lstm_sizes")
        self.activation: Callable = self.config.get("activation")

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
