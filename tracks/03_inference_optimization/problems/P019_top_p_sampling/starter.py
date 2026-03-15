import random
import math


def top_p_sample(logits, p: float):
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    probs = [(i, e / total) for i, e in enumerate(exps)]

    probs.sort(key=lambda x: x[1], reverse=True)

    cumulative = 0.0
    nucleus = []
    for idx, prob in probs:
        cumulative += prob
        nucleus.append((idx, prob))
        if cumulative >= p:
            break

    nucleus_total = sum(pr for _, pr in nucleus)
    r = random.random()
    cumulative = 0.0
    for idx, prob in nucleus:
        cumulative += prob / nucleus_total
        if r <= cumulative:
            return idx
    return nucleus[-1][0]
