# P006 Rotary Position Embedding (RoPE)

## Background

RoPE encodes position information by rotating pairs of dimensions in the query and key vectors. For each pair (x_{2i}, x_{2i+1}), a 2D rotation is applied:

```
[cos(mθ_i)  -sin(mθ_i)] [x_{2i}  ]
[sin(mθ_i)   cos(mθ_i)] [x_{2i+1}]
```

where m is the position index and θ_i = 10000^{-2i/d}.

Key advantages:
- **Relative position**: The dot product Q·K naturally encodes relative position (m-n)
- **Norm preservation**: Rotation preserves vector norms
- **Extrapolation**: With techniques like NTK-aware scaling or YaRN, RoPE extends to unseen lengths

## Interface Specification

```python
def apply_rope(
    x: torch.Tensor,       # (batch, seq_len, d_model), d_model must be even
    positions: torch.Tensor # (seq_len,) integer position indices
) -> torch.Tensor:          # (batch, seq_len, d_model)
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| Position 0 | θ angles are 0, so rotation is identity |
| Shape preserved | Output shape equals input shape |
| Norm preserved | ‖output‖ ≈ ‖input‖ for each vector |
| d_model even | Required for pairing dimensions |

## Practice Tips

1. Start by computing the frequency matrix θ
2. Apply rotation using the complex multiplication trick or explicit sin/cos
3. Verify position-0 gives identity and different positions give different outputs
