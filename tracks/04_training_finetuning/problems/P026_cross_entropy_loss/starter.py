import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, targets):
    return F.cross_entropy(logits, targets)
