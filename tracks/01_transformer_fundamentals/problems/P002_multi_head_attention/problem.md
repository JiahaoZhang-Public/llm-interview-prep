# P002 Multi-Head Attention

## Background

Multi-Head Attention (MHA) runs multiple attention functions in parallel, each on a different learned linear projection of Q, K, V. This allows the model to jointly attend to information from different representation subspaces.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Typical reshape flow: `(batch, seq, d_model)` → `(batch, num_heads, seq, head_dim)` → attention → `(batch, seq, d_model)`.

## Interface Specification

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int): ...
    def forward(
        self,
        query: torch.Tensor,   # (batch, seq_len, d_model)
        key: torch.Tensor,     # (batch, seq_len, d_model)
        value: torch.Tensor,   # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:         # (batch, seq_len, d_model)
```

## Example

```python
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
out = mha(x, x, x)  # shape: (2, 10, 64)
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| d_model % num_heads == 0 | Required for even splitting |
| num_heads = 1 | Degenerates to single-head attention |
| With mask | Mask applied identically across all heads |

## Practice Tips

1. Pay careful attention to reshape and transpose operations
2. Implement with separate W_Q, W_K, W_V, W_O projections
3. Verify output shape matches input shape
