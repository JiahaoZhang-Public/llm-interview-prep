import torch


def merge_lora_weights(base_weight, lora_a, lora_b, alpha: float = 1.0):
    rank = lora_a.size(0)
    scaling = alpha / rank
    return base_weight + lora_b @ lora_a * scaling
