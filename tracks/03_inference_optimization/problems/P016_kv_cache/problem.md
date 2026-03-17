# P016 Implement KV Cache

## Background

In Transformer autoregressive generation, producing each new token requires attending to **all previously generated tokens**.
Without caching, generating the t-th token requires recomputing Keys and Values for all t-1 prior positions, leading to O(n^3 d) overall complexity.

**KV Cache** caches the K_t, V_t computed at each step, appending them to the history. Each subsequent step only computes the new token's K, V, reducing total complexity to O(n^2 d).

```
Step 1: Q1 @ [K1]^T → attn → [V1] → output1          (cache = [K1, V1])
Step 2: Q2 @ [K1,K2]^T → attn → [V1,V2] → output2    (cache = [K1K2, V1V2])
Step 3: Q3 @ [K1,K2,K3]^T → attn → ...               (cache grows)
```

## Objective

Implement a minimal KV Cache class supporting incremental append and full retrieval.

## Interface

```python
class KVCache:
    def __init__(self):
        """Initialize empty cache"""

    def append(self, key, value):
        """Append one step's key and value to the cache"""

    def get(self):
        """Return (keys, values) — all cached K and V"""

    def reset(self):
        """Clear the cache (called on new prompt/request)"""
```

## Examples

```python
cache = KVCache()
cache.append([1.0, 2.0], [10.0, 20.0])
cache.append([3.0, 4.0], [30.0, 40.0])
keys, values = cache.get()
# keys == [[1.0, 2.0], [3.0, 4.0]], len(keys) == 2

cache.reset()
keys, values = cache.get()  # keys == [], values == []
```

## Constraints

| Condition | Description |
|-----------|-------------|
| Empty cache | `get()` should return two empty lists |
| After `reset()` | State returns to initialization |
| Multiple appends | Insertion order is preserved |
| key/value types | Python lists suffice; no torch Tensor required |

## Advanced Thinking

- Real model KV Cache shape: `(batch, num_heads, seq_len, head_dim)`
- Production systems pre-allocate a Tensor buffer and write by index, avoiding repeated concat
- Memory formula: `2 * num_layers * seq_len * num_heads * head_dim * dtype_bytes`

## Practice Tips

1. Implement with Python lists first, then pass all tests
2. Try rewriting with `torch.cat` (optional)
3. Verbally explain "why KV Cache reduces O(n^3) to O(n^2)"
