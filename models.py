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
        self.config = config

        if self.config["load_model"]:
            for agent_id, agent in self.agents.items():
                self = torch.load(self.config["load_model_from_path"])


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


class SpatialSoftMaxModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_shape": (100, 100),
            "num_actions": 5,
            "activation": "relu",
        }

        self.config = with_default_config(config, default_config)

        input_shape: Tuple[int, int] = self.config["input_shape"]
        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")

        self.activation: Callable = get_activation(self.config.get("activation"))


        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        layer_sizes = (input_size,) + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(layer_sizes[-1], num_actions)
        self.value_head = nn.Linear(layer_sizes[-1], 1)

    def soft_argmax(self, x:Tensor):
        n, c, h, w = x.size()

        posx, posy = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1., 1., w))
        posx = posx.reshape(h * w).to(x.device.type)
        posy = posy.reshape(h * w).to(x.device.type)

        x = x.reshape(-1, h * w)
        x = F.softmax(x, dim=-1)

        exp_x = torch.sum(posx * x, -1, keepdim=True)
        exp_y = torch.sum(posy * x, -1, keepdim=True)

        exp_xy = torch.cat([exp_x, exp_y], 1).to(x.device.type)
        return exp_xy.reshape(-1, c * 2)

    def indices(self, x:Tensor, x_only: bool = False):
        n, c, d, _ = x.size()
        m = x.view(n, c, -1).argmax(-1)
        indices = ((m // d).view(-1, 1), ) if x_only else ((m // d).view(-1, 1), (m % d).view(-1, 1))
        return torch.cat(indices, dim=-1)

    def forward(self, x: Tensor):
        # x = [n, 6, 100, 100]

        x1 = x[:, :3, :, :].to(x.device.type).to(x.device.type) # x1 = [n, 3, 100, 100]
        x2 = x[:, 3:, :, :].to(x.device.type).to(x.device.type) # x2 = [n, 3, 100, 100]

        x1 = self.conv(x1)
        x2 = self.conv(x2)

        x1 = self.soft_argmax(x1)
        x2 = self.soft_argmax(x2)

        x = torch.cat([x1, x2], dim=1)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value

    def get_initial_state(self):
        return ()


class BilinearCoordPooling(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_shape": (100, 100),
            "num_actions": 3,
            "activation": "relu",
            "field_threshold": 6,
            "hidden_sizes": (64, 64),

        }

        self.config = with_default_config(config, default_config)
        self.activation = get_activation(self.config["activation"])
        self.field_threshold = self.config["field_threshold"]

        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")
        input_shape: Tuple[int, int] = self.config["input_shape"]

        _coords_i = torch.linspace(-1, 1, input_shape[0]).view(-1, 1).repeat(1, input_shape[1])
        _coords_j = torch.linspace(-1, 1, input_shape[1]).view(1, -1).repeat(input_shape[0], 1)
        self.coords = torch.stack([_coords_i, _coords_j])

        self.bilinear = nn.Bilinear(2, 2, 4)
        self.pool1 = nn.AvgPool2d((100, self.field_threshold))
        self.pool2 = nn.AvgPool2d((100, 100-2*self.field_threshold))
        self.pool3 = nn.AvgPool2d((100, self.field_threshold))

        # concat + flatten to [B, 3*4]
        layer_sizes = (12,) + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(layer_sizes[-1], self.config["num_actions"])
        self.value_head = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x: Tensor):
        # noinspection PyTypeChecker

        batch_size = x.shape[0]
        batch_coords = torch.stack([self.coords for _ in range(batch_size)], dim=0).to(x.device.type)

        # transpose to [N, H, W, C]
        batch_coords = torch.transpose(batch_coords, -1, 1).contiguous()
        x = torch.transpose(x, -1, 1).contiguous()

        x = self.bilinear(x, batch_coords) # [N, 100, 100, 4]
        x = torch.transpose(x, -1, 1)

        # Pooling
        x1 = self.pool1(x[:,:,:,:self.field_threshold]).to(x.device.type)
        x2 = self.pool2(x[:,:,:,self.field_threshold:-self.field_threshold]).to(x.device.type)
        x3 = self.pool3(x[:,:,:,-self.field_threshold:]).to(x.device.type)

        x = torch.cat([x1, x2,x3], dim=1)
        x = x.flatten(1, -1)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Categorical(logits=action_logits)

        return action_distribution, value


if __name__ == '__main__':
    policy = MLPModel({})
    data = torch.randn(2, 15)

    action_dist, value_ = policy(data, ())
