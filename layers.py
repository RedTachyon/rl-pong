import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from typing import Dict, Any, Callable

from utils import with_default_config, get_activation


class RelationLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        default_config = {
            "num_subgoals": 2,
            "emb_size": 4,
            "rel_hiddens": (16, 16, ),
            "mlp_hiddens": (16, ),
            "activation": "leaky_relu"
        }

        self.config = with_default_config(config, default_config)

        self.activation: Callable[[Tensor], Tensor] = get_activation(self.config["activation"])

        self.own_embedding = nn.Parameter(torch.randn(self.config["emb_size"])/10., requires_grad=True)
        self.agent_embedding = nn.Parameter(torch.randn(self.config["emb_size"])/10., requires_grad=True)
        self.subgoal_embedding = nn.Parameter(torch.randn(self.config["emb_size"])/10., requires_grad=True)
        self.goal_embedding = nn.Parameter(torch.randn(self.config["emb_size"])/10., requires_grad=True)

        rel_sizes = (2 * (self.config["emb_size"] + 3), ) + self.config["rel_hiddens"]
        mlp_sizes = (self.config["rel_hiddens"][-1], ) + self.config["mlp_hiddens"]

        self.relation_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(rel_sizes, rel_sizes[1:])
        ])

        self.mlp_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(mlp_sizes, mlp_sizes[1:])
        ])

    def forward(self, x: Tensor):
        object_size = 3  # (x, y, flag)
        input_size = x.size()  # = (batch, [seq,] num_obj*3)
        num_objects = input_size[-1] // object_size
        x = x.view(input_size[:-1] + (num_objects, object_size))  # (batch, [seq,] num_obj, 3)
        # breakpoint()
        embeddings = torch.stack((self.own_embedding, self.agent_embedding)
                                 + (num_objects - 3) * (self.subgoal_embedding, )
                                 + (self.goal_embedding, ), dim=0)  # (num_obj, emb_size)

        # (1, [1, ] num_obj, emb_size)
        embeddings = embeddings.view(tuple(1 for _ in input_size[:-1]) + embeddings.size())

        # (batch, [seq, ] num_obj, emb_size)
        # embeddings = embeddings.expand(input_size[:-1] + (num_objects, self.config["emb_size"]))
        embeddings = embeddings.repeat(input_size[:-1] + (1, 1))
        inputs = torch.cat((x, embeddings), dim=-1)

        own_input = inputs[..., 0, :]  # (batch, [seq, ] emb_size+3)

        rel_outputs = 0
        for j in range(num_objects):
            other_input = inputs[..., j, :]  # (batch, [seq, ] emb_size+3)
            combined_input = torch.cat((own_input, other_input), dim=-1)

            intermediate_output = combined_input
            # noinspection PyTypeChecker
            for layer in self.relation_layers:
                intermediate_output = layer(intermediate_output)
                intermediate_output = self.activation(intermediate_output)

            rel_outputs += intermediate_output
        # breakpoint()
        # noinspection PyTypeChecker
        for layer in self.mlp_layers:
            rel_outputs = layer(rel_outputs)
            rel_outputs = self.activation(rel_outputs)

        return rel_outputs


if __name__ == '__main__':
    # Test of permutational invariance
    rellayer = RelationLayer({})
    inp1 = torch.rand(100, 21)
    inp2 = inp1.clone().detach()

    inp2[:, 6:9] = inp1[:, 9:12]
    inp2[:, 9:12] = inp1[:, 6:9]

    out1 = rellayer(inp1)
    out2 = rellayer(inp2)

    assert torch.allclose(out1, out2, atol=1e-5)
    assert not torch.allclose(inp1, inp2)

    rellayer_rec = RelationLayer({})

    inp_rec = torch.rand(100, 10, 21)
    out_rec = rellayer_rec(inp_rec)
