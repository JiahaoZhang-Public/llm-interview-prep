import torch
import math


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
