# P009 Transformer Decoder Layer

## Background

A Transformer Decoder Layer combines masked self-attention with a feed-forward network, forming the building block of decoder-only models (GPT, LLaMA, etc.).

The key difference from an encoder layer: the self-attention automatically applies a **causal mask** to prevent attending to future tokens.

```
x = x + MaskedSelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

In encoder-decoder architectures, the decoder layer also includes a cross-attention sublayer. For this problem, we focus on the decoder-only variant.

## Interface Specification

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int): ...
    def forward(
        self,
        x: torch.Tensor,       # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,  # if None, auto-generate causal mask
    ) -> torch.Tensor:          # (batch, seq_len, d_model)
```

## Constraints & Edge Cases

| Condition | Notes |
|-----------|-------|
| No mask provided | Automatically creates a causal mask |
| Explicit mask | Uses the provided mask instead |
| Single token | Works correctly with seq_len = 1 |
| Shape preserved | Output shape equals input shape |

## Practice Tips

1. Reuse `create_causal_mask` from P007 for automatic masking
2. Combine multi-head attention from P002 and FFN from P005
3. Use Pre-LN structure for stability
