# P022 Implement Repetition Penalty Logits Processor

## Background

**Repetition penalty** is a **logits processor** that modifies logits before sampling/generation to reduce the probability of already-seen tokens being selected again.

**Core formula (Keskar et al., 2019):**

For token i that has appeared before:
- If logit_i > 0: `logit_i = logit_i / penalty` (decreases)
- If logit_i < 0: `logit_i = logit_i * penalty` (more negative = lower)

Both operations reduce the effective probability of repeated tokens.

```
logits = [1.0, 2.0, 3.0]   generated_ids = [1]   penalty = 1.2
                                      ↓
logits[1] = 2.0 / 1.2 ≈ 1.67
result = [1.0, 1.67, 3.0]
```

**Why different handling for positive vs negative?**
- Positive logit → high probability → divide by penalty to lower it
- Negative logit → low probability → multiply by penalty to make it even lower (more negative)
- Goal is consistent: decrease the repeated token's probability

## Objective

Implement a repetition penalty logits processor.

## Interface

```python
def apply_repetition_penalty(
    logits: list[float],
    generated_ids: list[int],
    penalty: float
) -> list[float]:
    """Apply repetition penalty to logits of previously generated tokens"""
```

## Examples

```python
apply_repetition_penalty([1.0, 2.0, 3.0], [1], penalty=1.2)
# → [1.0, 1.667, 3.0]  (only index 1 is penalized)

apply_repetition_penalty([1.0, -2.0, 3.0], [1], penalty=1.2)
# → [1.0, -2.4, 3.0]   (negative logit multiplied by penalty)

apply_repetition_penalty([1.0, 2.0, 3.0], [], penalty=1.2)
# → [1.0, 2.0, 3.0]    (no history, no modification)
```

## Constraints

| Condition | Description |
|-----------|-------------|
| penalty | >= 1.0 (1.0 = no effect, higher = stronger penalty) |
| generated_ids | May be empty, may have duplicates (deduplicate) |
| token_id range | 0 <= id < len(logits) |
| Immutability | Return a new list, don't modify the original |

## Practice Tips

1. Implement basic logic: iterate over generated_ids, handle positive/negative separately
2. Deduplicate: same token appearing multiple times only needs one penalty application (repetition vs frequency penalty)
3. Think about the difference between frequency penalty and repetition penalty
