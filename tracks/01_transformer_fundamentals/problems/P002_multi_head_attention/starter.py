import torch
import math
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)
