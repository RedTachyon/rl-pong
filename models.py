from typing import Dict, Tuple, Callable, Iterable

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Categorical
from torch.nn import functional as F

from layers import RelationLayer
from utils import with_default_config, get_activation, initialize_unit_


class BaseModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]:
        raise NotImplementedError


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

    def forward(self, x: Tensor) -> Tuple[Distribution, Tensor]:
        x = x.view((x.shape[0], -1))
        # noinspection PyTypeChecker
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value


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

        self.conv_layers = nn.ModuleList([nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 24x24x32
                                          nn.Conv2d(32, 64, kernel_size=7, stride=3),  # 6x6x64
                                          nn.Conv2d(64, 64, kernel_size=3, stride=1)])  # 4x4x64

        _coords_i = torch.linspace(-1, 1, input_shape[0]).view(-1, 1).repeat(1, input_shape[1])
        _coords_j = torch.linspace(-1, 1, input_shape[1]).view(1, -1).repeat(input_shape[0], 1)
        self.coords = torch.stack([_coords_i, _coords_j])

        # flatten

        self.policy_head = nn.Linear(4*4*64, self.config["num_actions"])
        self.value_head = nn.Linear(4*4*64, 1)

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        batch_coords = torch.stack([self.coords for _ in range(batch_size)], dim=0)
        batch_coords = batch_coords.to(x.device.type)
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

        return action_distribution, value


class BilinearCoordPooling(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_shape": (100, 100),
            "num_actions": 3,
            "activation": "relu",
            "field_threshold": 6,

        }

        self.config = with_default_config(config, default_config)
        self.activation = get_activation(self.config["activation"])
        self.field_threshold = self.config["field_threshold"]

        input_shape: Tuple[int, int] = self.config["input_shape"]

        _coords_i = torch.linspace(-1, 1, input_shape[0]).view(-1, 1).repeat(1, input_shape[1])
        _coords_j = torch.linspace(-1, 1, input_shape[1]).view(1, -1).repeat(input_shape[0], 1)
        self.coords = torch.stack([_coords_i, _coords_j])

        self.bilinear = nn.Bilinear(2, 2, 4, bias=False)
        initialize_unit_(self.bilinear)

        self.pool = nn.AvgPool2d(6, stride=6)

        # flatten
        self.linear_layers = nn.ModuleList([nn.Linear(1024, 16, bias=False),
                                            nn.Linear(16, 16),
                                            nn.Linear(16, 16)])

        self.policy_head = nn.Linear(16, self.config["num_actions"])
        self.value_head = nn.Linear(16, 1)

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        batch_coords = torch.stack([self.coords for _ in range(batch_size)], dim=0).to(x.device.type)

        # transpose to [N, H, W, C]
        batch_coords = torch.transpose(batch_coords, 1, -1).contiguous()
        x = torch.transpose(x, -1, 1).contiguous()

        x = self.bilinear(x, batch_coords)  # [N, 100, 100, 4]
        x = torch.transpose(x, -1, 1)

        # breakpoint()

        # Pool
        x = self.pool(x)
        x = torch.flatten(x, 1)

        # noinspection PyTypeChecker
        for layer in self.linear_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value


if __name__ == '__main__':
    pass
    # policy = MLPModel({})
    # data = torch.randn(2, 15)
    #
    # action_dist, value_ = policy(data, ())
