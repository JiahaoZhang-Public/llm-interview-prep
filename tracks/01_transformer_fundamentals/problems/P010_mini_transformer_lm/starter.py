import torch
from torch import nn


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        raise NotImplementedError("Implement P010 in this starter.")

    def forward(self, input_ids):
        raise NotImplementedError("Implement P010 in this starter.")
