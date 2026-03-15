import random
import math


def temperature_sample(logits, temperature: float = 1.0):
    scaled = [l / temperature for l in logits]
    max_val = max(scaled)
    exps = [math.exp(s - max_val) for s in scaled]
    total = sum(exps)
    probs = [e / total for e in exps]

    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1
