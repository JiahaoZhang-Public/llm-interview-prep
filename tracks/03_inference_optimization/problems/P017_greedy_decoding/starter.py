import math


def greedy_decode(logits):
    max_val = logits[0]
    max_idx = 0
    for i, val in enumerate(logits):
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx
