# P001 Scaled Dot-Product Attention

## Background

Scaled Dot-Product Attention is the core computation in the Transformer architecture. Given Query (Q), Key (K), and Value (V) matrices, the formula is:

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

Key insights:
- **Why scale by √d_k?** As the dimension d_k grows, the dot products grow in magnitude, pushing the softmax into regions with extremely small gradients. Dividing by √d_k keeps the variance of the dot products at ~1.
- **Mask**: In autoregressive generation, a causal mask prevents attending to future tokens; in encoder tasks, a padding mask zeroes out padding positions.

## Interface Specification

```python
def scaled_dot_product_attention(
    query: torch.Tensor,    # (batch, seq_len_q, d_k)
    key: torch.Tensor,      # (batch, seq_len_k, d_k)
    value: torch.Tensor,    # (batch, seq_len_k, d_v)
    mask: Optional[torch.Tensor] = None,  # broadcastable to (batch, seq_len_q, seq_len_k)
) -> torch.Tensor:          # (batch, seq_len_q, d_v)
```

## Example

```python
Q = torch.randn(1, 3, 8)
K = torch.randn(1, 3, 8)
V = torch.randn(1, 3, 8)
out = scaled_dot_product_attention(Q, K, V)  # shape: (1, 3, 8)
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| No mask | All positions attend to all positions |
| Causal mask | Upper triangle filled with `-inf` |
| d_k = 1 | Scaling still applies |
| Single token | Q has seq_len_q = 1, output shape matches |
| Attention weights | Each row of softmax output sums to 1 |

## Practice Tips

1. Implement step by step: scores → scale → mask → softmax → weighted sum
2. Verify that attention weights sum to 1 along the key dimension
3. Be prepared to explain why √d_k scaling matters for training stability
