# P021 Implement Beam Search

## Background

**Beam search** is an approximate search algorithm — a middle ground between greedy (keeping 1 path) and exhaustive search (keeping all paths).

**Core idea:** At each step, retain the `beam_size` candidate sequences with the highest cumulative scores, rather than only keeping the current best (greedy). This allows correcting sub-optimal early choices in later steps.

```
beam_size = 2, vocab = {A, B, C}

Step 0:  [START]
Step 1:  [START, A] score=-0.2     ← keep
         [START, B] score=-0.5     ← keep
         [START, C] score=-1.0     ← discard
Step 2:  [START, A, B] score=-0.3  ← keep
         [START, B, A] score=-0.6  ← keep
         ...
```

**Key properties:**
- beam_size=1 degenerates to greedy decoding
- Beam search is **deterministic** (unlike sampling)
- Uses **log probability** accumulation to avoid probability underflow
- Must handle EOS: finished sequences stop expanding but remain in candidate ranking

## Objective

Implement a general-purpose beam search function.

## Interface

```python
def beam_search(
    step_fn,              # Function: given sequence → [(log_prob, token), ...]
    start_tokens,         # Initial token sequence (list)
    beam_size: int,       # Beam width
    max_new_tokens: int,  # Maximum new tokens to generate
    eos_token_id=None     # End-of-sequence token id (optional)
) -> list:
    """Return the highest-scoring complete sequence"""
```

## Examples

```python
def step_fn(tokens):
    last = tokens[-1]
    if last == 2: return [(-0.1, 2)]
    return [(-0.1, last + 1), (-0.5, 2)]

seq = beam_search(step_fn, [0], beam_size=2, max_new_tokens=3, eos_token_id=2)
# Returns the highest-scoring sequence starting with [0, ...]
```

## Constraints

| Condition | Description |
|-----------|-------------|
| beam_size | >= 1 |
| EOS handling | Beams ending with EOS stop expanding but remain in candidate pool |
| All EOS | If all beams end with EOS, stop early |
| Return value | The top-scoring sequence (includes start_tokens) |
| Score type | Log probability, higher is better (all negative or zero) |

## Practice Tips

1. Implement without EOS handling first
2. Add EOS early stopping
3. Verbally explain beam search vs greedy vs sampling tradeoffs
4. Think about length normalization and why it's needed
