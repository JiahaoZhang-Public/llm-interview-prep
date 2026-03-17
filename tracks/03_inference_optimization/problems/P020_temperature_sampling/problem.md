# P020 Implement Temperature Sampling

## Background

**Temperature** is the most fundamental hyperparameter for controlling LLM generation randomness.

The principle is simple: divide logits by temperature T before applying softmax:

$$\text{softmax}(z_i / T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**Temperature effects:**
- **T < 1** (low): Amplifies logit differences → distribution becomes **sharper** → more deterministic, conservative
- **T = 1**: Original distribution, no change
- **T > 1** (high): Shrinks logit differences → distribution becomes **flatter** → more random, creative
- **T → 0**: Degenerates to greedy decoding
- **T → ∞**: Degenerates to uniform distribution

```
logits = [2.0, 1.0, 0.5]

T=0.5:  probs ≈ [0.84, 0.11, 0.04]   ← more deterministic
T=1.0:  probs ≈ [0.59, 0.24, 0.17]   ← original
T=2.0:  probs ≈ [0.43, 0.31, 0.26]   ← more random
```

## Objective

Implement pure Python temperature sampling.

## Interface

```python
def temperature_sample(logits: list[float], temperature: float = 1.0) -> int:
    """Scale logits by temperature, then sample"""
```

## Examples

```python
import random
random.seed(0)
temperature_sample([1.0, 2.0, 3.0], temperature=0.7)  # → more likely 2
temperature_sample([1.0, 2.0, 3.0], temperature=5.0)  # → all three have similar odds
```

## Constraints

| Condition | Description |
|-----------|-------------|
| temperature | > 0 (handling T=0 not required) |
| Numerical stability | Subtract max(scaled_logits) before exp |
| Randomness | Use `random` module |

## Practice Tips

1. Implement the full pipeline: logits / T → softmax → sample
2. Manually verify: does low T converge toward greedy?
3. Verbally explain why T is "divided" not "multiplied"
