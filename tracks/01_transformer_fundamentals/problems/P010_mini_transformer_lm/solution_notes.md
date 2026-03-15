# P010 Mini Transformer LM — Solution Notes

## Core Approach

1. **Token + Position Embedding**: Token embedding lookup + learnable position embedding
2. **Transformer Block**: Causal self-attention → LayerNorm → FFN → LayerNorm (Pre-LN or Post-LN both work)
3. **LM Head**: Final Linear layer maps hidden_size → vocab_size to produce logits
4. **Causal Mask**: Upper-triangular mask ensures autoregressive property — position i can only attend to positions ≤ i

## Interview Oral Template

> "A minimal Transformer LM has three parts: an embedding layer, a transformer block, and an LM head.
> The embedding layer maps token IDs and positions to hidden_size vectors and sums them.
> The transformer block first does causal self-attention with an upper-triangular mask to enforce
> autoregressive ordering, followed by residual + LayerNorm. Then a two-layer FFN
> (expand → GELU → project back) with another residual + LayerNorm.
> The LM head is a Linear(hidden_size → vocab_size) that outputs next-token logits at each position."

## Common Pitfalls

- **Causal mask direction reversed**: Future positions should be masked (upper triangle = True/blocked)
- **Position embedding length too short**: Either use fixed length (e.g., 512) or relative position encodings like RoPE
- **Missing residual connections**: Attention/FFN inputs must be added back to outputs
- **Pre-LN vs Post-LN**: Be explicit about which variant you're using; Pre-LN is more stable and common in modern models

## Complexity

- Time: O(n² · d + n · d · d_ff), attention is the n² bottleneck
- Space: O(n² + n · d)
- Parameters: ~4d² + 2d · d_ff + vocab_size · d (single layer)
