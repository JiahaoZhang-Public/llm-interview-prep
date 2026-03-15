import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0, bias: bool = False):
        super().__init__()
        raise NotImplementedError("Implement P029 in this starter.")

    def forward(self, x):
        raise NotImplementedError("Implement P029 in this starter.")
