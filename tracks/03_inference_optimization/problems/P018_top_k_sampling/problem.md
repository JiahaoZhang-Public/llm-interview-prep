# P018 Implement Top-K Sampling

## Background

**Top-K sampling** is a constrained random sampling strategy: keep only the K tokens with the highest probabilities, zero out the rest, renormalize within the top-K set, and sample randomly.

```
logits = [1.0, 5.0, 4.5, 0.1, 0.2]    (vocab_size = 5)
                ↓ k=2
Keep index 1 (5.0) and index 2 (4.5)
                ↓ softmax on [5.0, 4.5]
probs ≈ [0.62, 0.38]
                ↓ sample by probability
Return index 1 or 2
```

**Why Top-K?**
- Pure softmax sampling can produce garbage by sampling from long-tail low-probability tokens
- Top-K truncates the distribution tail, balancing diversity and quality
- K=1 degenerates to greedy decoding

**Top-K's limitation:** K is fixed, but probability distributions vary widely across positions.
When the distribution is sharp, K=50 is too many; when flat, K=50 is too few. This motivates Top-P sampling.

## Objective

Implement pure Python top-k sampling.

## Interface

```python
def top_k_sample(logits: list[float], k: int) -> int:
    """Sample randomly from the top-k candidates"""
```

## Examples

```python
import random
random.seed(0)
top_k_sample([0.1, 0.9, 0.8, 0.0], k=2)  # → 1 or 2 (only top-2)
top_k_sample([0.1, 0.9, 0.8, 0.0], k=1)  # → 1 (k=1 = greedy)
```

## Constraints

| Condition | Description |
|-----------|-------------|
| k range | 1 <= k <= len(logits) |
| Return value | Must be an original index within the top-k |
| Numerical stability | Subtract max logit before exp to prevent overflow |
| Randomness | Use `random` module; controllable via `random.seed` |

## Practice Tips

1. Sort → take top-k → softmax → sample, implement step by step
2. Pay special attention to numerical stability (subtract-max trick)
3. Verbally compare top-k vs top-p pros and cons
