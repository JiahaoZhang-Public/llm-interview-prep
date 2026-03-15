import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        raise NotImplementedError("Implement P005 in this starter.")

    def forward(self, x):
        raise NotImplementedError("Implement P005 in this starter.")
