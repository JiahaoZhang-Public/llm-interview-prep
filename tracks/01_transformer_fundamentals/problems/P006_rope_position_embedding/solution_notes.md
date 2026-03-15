# P006 Rotary Position Embedding (RoPE) — Solution Notes

## Core Approach

1. RoPE encodes position by rotating pairs of dimensions in Q and K vectors
2. For position m, pair dimensions (2i, 2i+1) are rotated by angle m · theta_i
3. theta_i = 1 / (10000^(2i/d)) — lower-frequency rotations for higher dimension pairs
4. Rotation is applied as: [q_2i, q_2i+1] → [q_2i·cos(mθ) - q_2i+1·sin(mθ), q_2i·sin(mθ) + q_2i+1·cos(mθ)]
5. Key property: the dot product Q_m · K_n depends only on relative position (m - n)

## Interview Oral Template

> "RoPE encodes absolute position but yields relative position information
> in the attention scores. It works by grouping dimensions into pairs and
> applying a 2D rotation to each pair, where the rotation angle is the product
> of the token position and a frequency that decreases with dimension index.
> After rotation, the dot product between query at position m and key at
> position n only depends on m minus n, giving us relative position awareness.
> RoPE is popular in LLaMA and other modern models because it has no learnable
> parameters, extrapolates well, and decays attention naturally with distance."

## Common Pitfalls

- **Applying RoPE to V**: RoPE is only applied to Q and K, never to V — V carries content, not positional information
- **Wrong dimension pairing**: Must pair consecutive dimensions (0,1), (2,3), etc. — not (0, d/2)
- **Frequency computation**: theta_i uses 2i/d not i/d; getting this wrong changes the frequency spectrum
- **Not precomputing sin/cos tables**: For efficiency, precompute the rotation matrices for all positions up to max_seq_len

## Complexity

- Time: O(n · d) — just element-wise multiply and add for each position
- Space: O(n · d) for precomputed sin/cos tables (or O(d) if computed on the fly)
- No learnable parameters — purely geometric encoding
