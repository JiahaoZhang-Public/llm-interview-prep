# P010 Mini Transformer Language Model

## Background

This problem assembles all previous components into a complete autoregressive language model:

```
Input token IDs
    → Token Embedding + Position Encoding
    → N × Transformer Decoder Layers
    → Final LayerNorm
    → Linear (lm_head) → vocab logits
```

Key design choices:
- **Weight tying**: The embedding matrix and lm_head can share weights, reducing parameters
- **Position encoding**: Can use learned embeddings, sinusoidal, or RoPE
- **Output**: Raw logits (unnormalized) over the vocabulary

## Interface Specification

```python
class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int = 512,
    ): ...
    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len) long tensor
    ) -> torch.Tensor:            # (batch, seq_len, vocab_size)
```

## Example

```python
model = MiniTransformerLM(vocab_size=1000, d_model=64, num_heads=4, d_ff=256, num_layers=2)
ids = torch.randint(0, 1000, (2, 10))
logits = model(ids)  # shape: (2, 10, 1000)
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| Output dimension | Last dim = vocab_size |
| Single token | Works with seq_len = 1 |
| Different vocab sizes | Output dim changes accordingly |
| Gradient flow | All parameters should receive gradients |

## Practice Tips

1. Stack all components: Embedding → DecoderLayers → LN → lm_head
2. Ensure causal masking is handled inside decoder layers
3. Verify gradients flow through the entire model with a simple backward pass
