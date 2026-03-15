import math
import torch


def clip_gradients(parameters, max_norm: float):
    parameters = list(parameters)
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return total_norm.item()
