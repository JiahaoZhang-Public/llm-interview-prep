import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True):
        super().__init__()
        raise NotImplementedError("Implement P002 in this starter.")

    def forward(self, x, mask=None):
        raise NotImplementedError("Implement P002 in this starter.")
