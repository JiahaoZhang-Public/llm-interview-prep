# P001 Scaled Dot-Product Attention — Solution Notes

## Core Approach

1. Compute Q·K^T to get raw attention scores
2. Divide by √d_k to prevent softmax saturation when d_k is large
3. If mask is provided, add it (use -inf to zero out positions after softmax)
4. Apply softmax along the key dimension, then multiply by V

Key formula: `Attention(Q,K,V) = softmax(QK^T / √d_k) · V`

## Interview Oral Template

> "Scaled dot-product attention first computes raw scores via Q times K-transpose,
> then divides by square root of d_k — this scaling prevents the dot products from
> growing too large when d_k is big, which would push softmax into regions with
> near-zero gradients. The mask is applied additively (using -inf for blocked positions),
> so after softmax those positions become zero. Finally, we multiply the attention
> weights by V to get the weighted sum of values."

## Common Pitfalls

- **Forgetting to scale**: Without dividing by √d_k, softmax saturates at high dimensions, causing vanishing gradients
- **Using multiplicative mask instead of additive**: Multiplying by 0 gives 0 before softmax (not -inf), so it still gets some probability
- **Wrong transpose dimensions**: K must be transposed on the last two dims (-2, -1), not the entire tensor
- **Softmax on wrong axis**: Must be applied on the last dimension (key sequence length)

## Complexity

- Time: O(n² · d), where n is sequence length, d is head dimension
- Space: O(n²) for the attention weight matrix
- This quadratic cost is why long sequences need FlashAttention or similar IO-aware optimizations
