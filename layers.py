import torch
from torch import nn

from typing import Dict, Any


class RelationLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # TODO: Write the relation layer