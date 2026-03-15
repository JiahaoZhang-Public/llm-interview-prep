import random
import math


def top_k_sample(logits, k: int):
    indexed = list(enumerate(logits))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_k = indexed[:k]

    max_logit = max(v for _, v in top_k)
    exps = [(idx, math.exp(v - max_logit)) for idx, v in top_k]
    total = sum(e for _, e in exps)
    probs = [(idx, e / total) for idx, e in exps]

    r = random.random()
    cumulative = 0.0
    for idx, p in probs:
        cumulative += p
        if r <= cumulative:
            return idx
    return probs[-1][0]
