# P019 Implement Top-P (Nucleus) Sampling

## Background

**Top-P sampling** (aka **nucleus sampling**, Holtzman et al., 2020) is an adaptive sampling strategy.
Unlike Top-K's fixed candidate count, Top-P dynamically determines the candidate set size based on cumulative probability.

**Algorithm:**
1. Compute softmax probabilities from logits
2. Sort probabilities descending
3. Accumulate from the largest until cumulative probability >= p
4. These tokens form the **nucleus**
5. Renormalize within the nucleus and sample

```
probs (sorted) = [0.50, 0.25, 0.15, 0.07, 0.03]
                  ↓ p=0.8
cumsum =          [0.50, 0.75, 0.90, ...]
                              ↑ reaches >= 0.8 at 3rd token
nucleus = {token_0, token_1, token_2}   (dynamic size = 3)
```

**Top-P advantages:**
- When distribution is sharp, nucleus is small (near-greedy); when flat, nucleus is large (preserves diversity)
- p=1.0 = full distribution sampling; p→0 = greedy
- More adaptive than fixed K

## Objective

Implement pure Python top-p (nucleus) sampling.

## Interface

```python
def top_p_sample(logits: list[float], p: float) -> int:
    """Sample from the smallest candidate set with cumulative probability >= p"""
```

## Examples

```python
import random
random.seed(0)
top_p_sample([3.0, 2.0, 0.1], p=0.8)   # → 0 or 1 (their probabilities sum >= 0.8)
top_p_sample([100.0, 0.0, 0.0], p=0.5)  # → 0 (extremely sharp, only 1 candidate)
```

## Constraints

| Condition | Description |
|-----------|-------------|
| p range | 0 < p <= 1.0 |
| Sort direction | Must accumulate from highest probability |
| Renormalization | Must renormalize within nucleus before sampling |
| Boundary inclusion | Token where cumsum first reaches >= p is included |

## Practice Tips

1. Implement step by step: softmax → sort → cumsum → truncate → renormalize → sample
2. Consider degenerate cases: p=1.0 and p→0
3. Verbally explain the core difference between top-k and top-p
