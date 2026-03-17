# P017 Implement Greedy Decoding

## Background

**Greedy decoding** is the simplest text generation strategy: at each step, select the token with the highest probability (i.e., argmax of the logits).

```
logits = [0.1, 0.9, 0.2, 0.05]
         ↓
argmax → index 1
```

While simple, efficient, and **fully deterministic**, greedy has a fundamental flaw:
**locally optimal ≠ globally optimal**. Greedily choosing the best token at each step may lead to poor overall sequence quality, and it's prone to **repetition degeneration**.

This is why advanced strategies like beam search, top-k, and top-p sampling exist.

## Objective

Implement a pure Python greedy decode function: given a logits list, return the index of the maximum value.

## Interface

```python
def greedy_decode(logits: list[float]) -> int:
    """Return the index of the maximum value in logits"""
```

## Examples

```python
greedy_decode([0.1, 0.9, 0.2])  # → 1
greedy_decode([5.0, 5.0, 1.0])  # → 0 (ties: return first)
greedy_decode([-1.0, -2.0])     # → 0 (negative values work fine)
```

## Constraints

| Condition | Description |
|-----------|-------------|
| logits length | >= 1 |
| Value range | Any float, including negatives |
| Tie-breaking | Return the first index when multiple maxima exist |
| No softmax needed | Operate directly on raw logits |

## Practice Tips

1. Write the simplest traversal version first
2. Think: Why is softmax unnecessary? (softmax is a monotonically increasing transform)
3. Verbally compare greedy vs beam search vs sampling tradeoffs
