# P002 Multi-Head Attention — Solution Notes

## Core Approach

1. Project input through three linear layers to get Q, K, V
2. Split hidden_size into num_heads × head_dim, reshape + transpose to (B, H, T, D)
3. Each head independently performs scaled dot-product attention
4. Concatenate all heads' outputs, apply output projection

Key insight: Multiple heads let the model attend to different representation subspaces simultaneously.

## Interview Oral Template

> "Multi-head attention captures different representation subspaces through parallel attention heads.
> We split hidden_size into H heads, each with dimension d_k = hidden_size / H.
> First, we project through W_Q, W_K, W_V, then reshape to (B, H, T, d_k).
> Each head does scaled dot-product attention independently.
> We concatenate back to (B, T, hidden_size) and apply an output projection W_O.
> The output projection is crucial — it's where information from different heads gets fused."

## Common Pitfalls

- **head_dim calculation error**: hidden_size must be divisible by num_heads
- **Wrong reshape/transpose order**: First view as (B,T,H,D), then transpose(1,2) to get (B,H,T,D)
- **Missing contiguous() before view**: After transpose, memory layout is non-contiguous
- **Forgetting output projection**: The concat output must go through W_O for cross-head information mixing

## Complexity

- Time: O(n² · d), same as single-head (just split into smaller parallel heads)
- Space: O(H · n²) for attention weights across all heads
- Parameters: 3 × d² + d² (QKV projections + output projection)
