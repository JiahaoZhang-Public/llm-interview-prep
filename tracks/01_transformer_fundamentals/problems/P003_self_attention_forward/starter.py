import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        raise NotImplementedError("Implement P003 in this starter.")

    def forward(self, x, mask=None):
        raise NotImplementedError("Implement P003 in this starter.")
