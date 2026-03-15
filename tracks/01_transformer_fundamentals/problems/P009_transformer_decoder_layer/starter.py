import torch
from torch import nn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        raise NotImplementedError("Implement P009 in this starter.")

    def forward(self, x, mask=None):
        raise NotImplementedError("Implement P009 in this starter.")
