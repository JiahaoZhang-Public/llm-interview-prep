# P007 Causal Mask

## Background

In autoregressive (decoder-only) models, each token should only attend to itself and previous tokens. The causal mask enforces this by adding `-inf` to the upper-triangular portion of the attention scores:

```
Mask (4×4):
  0    -inf  -inf  -inf
  0     0    -inf  -inf
  0     0     0    -inf
  0     0     0     0
```

After adding this mask to the raw scores and applying softmax, the `-inf` entries become 0, effectively blocking future positions.

## Interface Specification

```python
def create_causal_mask(seq_len: int) -> torch.Tensor:
    # Returns (seq_len, seq_len) float tensor
    # Lower triangle + diagonal = 0.0
    # Upper triangle = -inf
```

## Example

```python
mask = create_causal_mask(3)
# tensor([[  0., -inf, -inf],
#         [  0.,   0., -inf],
#         [  0.,   0.,   0.]])
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| seq_len = 1 | Returns [[0.0]] |
| Diagonal | All zeros (each token attends to itself) |
| Lower triangle | All zeros (attend to past) |
| Upper triangle | All -inf (block future) |
| Additive mask | Added to scores before softmax |

## Practice Tips

1. Use `torch.triu` with diagonal=1 to select the upper triangle
2. Fill selected positions with `float('-inf')`
3. Understand the difference between boolean masks and additive masks
