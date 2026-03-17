# P005 Position-wise Feed-Forward Network

## Background

The Position-wise FFN applies the same two-layer MLP independently to each position:

```
FFN(x) = activation(x W_1 + b_1) W_2 + b_2
```

Key properties:
- **Position-wise**: The same weights are shared across all positions (equivalent to two 1×1 convolutions)
- **Expansion ratio**: Typically d_ff = 4 × d_model, creating a bottleneck architecture
- **Activation evolution**: ReLU (original) → GELU (GPT/BERT) → SwiGLU (LLaMA), where SwiGLU uses a gating mechanism

## Interface Specification

```python
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) → (batch, seq_len, d_model)
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| Expansion | d_ff > d_model (typically 4×) |
| Nonlinear | Output differs from a simple linear transform |
| Single token | Works correctly with seq_len = 1 |

## Practice Tips

1. Implement with `nn.Linear(d_model, d_ff)` → activation → `nn.Linear(d_ff, d_model)`
2. Compare ReLU vs GELU vs SwiGLU in terms of parameter count and expressiveness
3. Understand why this is called "position-wise" (same MLP at every position)
