# P004 Transformer Block

## Background

A Transformer Block is the fundamental building unit, stacking self-attention and feed-forward network with residual connections and layer normalization.

**Pre-LN (modern default)**:
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Post-LN (original paper)**:
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

Pre-LN is preferred in practice because it stabilizes training for deep models without careful learning rate warmup.

## Interface Specification

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int): ...
    def forward(
        self,
        x: torch.Tensor,       # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:          # (batch, seq_len, d_model)
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| Residual connection | Output = input + sublayer(input) |
| Contains LayerNorm | At least 2 LayerNorm layers |
| d_ff configurable | FFN intermediate dimension independent of d_model |

## Practice Tips

1. Draw the data flow diagram: LN → Attention → Add → LN → FFN → Add
2. Verify the residual connection: set sublayer weights to zero and check output ≈ input
3. Count parameters: attention (4 × d² ) + FFN (2 × d × d_ff) + norms
