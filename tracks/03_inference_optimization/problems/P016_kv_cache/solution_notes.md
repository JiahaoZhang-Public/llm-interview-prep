# P016 KV Cache — Solution Notes

## Core Approach

1. During autoregressive generation, each step produces one new token's Q but needs attention against all previous K and V
2. KV Cache stores previously computed K and V vectors; each step only computes the new token's K, V and appends them
3. This reduces per-step attention from O(n²d) to O(nd), and total generation from O(n³d) to O(n²d)

## Interview Oral Template

> "KV Cache is the essential optimization for autoregressive inference.
> Without it, generating token t requires recomputing K and V for all t previous positions — O(n³) total.
> With KV Cache, we store each step's key and value vectors and append the new ones.
> The next step only computes q, k, v for the new token, then attends against the full cached K.
> Per-step cost drops from O(n²) to O(n), total from O(n³) to O(n²).
> The tradeoff is extra memory: each layer stores 2 × n × d floats."

## Common Pitfalls

- **Forgetting to reset**: Each new prompt must clear the cache, otherwise context from the previous prompt leaks
- **Per-layer caches**: Real models have independent KV caches for every transformer layer
- **Memory estimation**: KV Cache size = 2 × num_layers × seq_len × hidden_size × dtype_bytes
  - Example: 70B model, 80 layers, 8192 seq_len, fp16: ~20GB
- **Concat vs pre-allocation**: Repeated tensor concatenation is slow; production systems pre-allocate fixed buffers and write by index

## Complexity

- append: O(1) amortized
- get: O(1)
- Memory: O(n × d) per layer, where n is generated length
