# P009 Transformer Decoder Layer — Solution Notes

## Core Approach

1. Masked multi-head self-attention with built-in causal mask — prevents attending to future positions
2. Add & Norm: residual connection + LayerNorm after self-attention
3. Feed-forward network: position-wise two-layer MLP
4. Add & Norm: residual connection + LayerNorm after FFN
5. The causal mask is generated internally (no external mask needed for standard autoregressive use)
6. This is the GPT-style decoder (no cross-attention to encoder)

## Interview Oral Template

> "A decoder-only Transformer layer is like an encoder block but with a causal
> mask built into the self-attention. The causal mask ensures each position
> can only attend to itself and earlier positions, which is what makes it
> autoregressive. The flow is: masked self-attention with residual and LayerNorm,
> then a position-wise FFN with residual and LayerNorm. For a full encoder-decoder
> architecture like the original Transformer, there's also a cross-attention
> sublayer between self-attention and FFN, but GPT-style decoders skip that."

## Common Pitfalls

- **Forgetting the causal mask**: Without it, the decoder can cheat by looking at future tokens during training
- **Mask not applied during training**: The mask must be present during training even though teacher forcing provides all tokens at once
- **Pre-LN vs Post-LN confusion**: Know which variant is expected — Pre-LN (norm before sublayer) is more stable for deep models
- **Confusing with encoder-decoder**: GPT-style decoders have no cross-attention; full Transformer decoders have three sublayers (self-attn, cross-attn, FFN)

## Complexity

- Time: O(n^2 · d + n · d · d_ff), same as encoder block but with masked attention
- Space: O(n^2) for causal attention matrix
- During inference with KV-cache, per-step cost drops to O(n · d) since only the new token's row is computed
