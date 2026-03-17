# P008 Layer Normalization

## Background

Layer Normalization normalizes across the feature dimension (last dimension) for each sample independently:

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

where μ and σ² are the mean and variance computed over the last dimension, γ (weight) and β (bias) are learnable affine parameters.

Key differences from Batch Normalization:
- **LN** normalizes across features, independent of batch → works with batch_size=1 and variable-length sequences
- **BN** normalizes across the batch → requires large batches and fixed dimensions

Important: PyTorch's `LayerNorm` uses `unbiased=False` (divides by N, not N-1) for the variance calculation.

## Interface Specification

```python
def layer_norm(
    x: torch.Tensor,                    # (..., d)
    weight: torch.Tensor,               # (d,)
    bias: torch.Tensor,                 # (d,)
    eps: float = 1e-5,
) -> torch.Tensor:                      # same shape as x
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| unbiased=False | Variance uses N denominator, not N-1 |
| eps | Prevents division by zero |
| Affine | weight (γ) and bias (β) applied element-wise |
| Shape preserved | Output shape identical to input |
| Matches PyTorch | Should produce same results as `nn.LayerNorm` |

## Practice Tips

1. Compute mean and variance along the last dimension with `keepdim=True`
2. Remember to use `unbiased=False` (or equivalently, use `.var(correction=0)`)
3. Compare your output with `torch.nn.LayerNorm` to verify correctness
