# P003 Self-Attention Forward Pass

## Background

Self-Attention is the specific case of attention where Q, K, V all come from the **same** input sequence, with learned linear projections:

```
Q = X W_Q,  K = X W_K,  V = X W_V
output = Attention(Q, K, V)
```

This differs from:
- P001 (raw attention, no projections)
- P002 (multi-head with reshape/split)

Here we implement **single-head** self-attention with W_Q, W_K, W_V projections.

## Interface Specification

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model: int): ...
    def forward(
        self,
        x: torch.Tensor,       # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:          # (batch, seq_len, d_model)
```

## Example

```python
sa = SelfAttention(d_model=64)
x = torch.randn(2, 10, 64)
out = sa(x)  # shape: (2, 10, 64)
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| Single token | seq_len = 1, output equals projected value |
| No mask | All positions attend to each other |
| Three projections | W_Q, W_K, W_V must exist as parameters |

## Practice Tips

1. Ensure Q, K, V are all derived from the same input X
2. Use `scaled_dot_product_attention` from P001 internally
3. Explain the difference between self-attention and cross-attention
