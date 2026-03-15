# P004 Transformer Block — Solution Notes

## Core Approach

1. Multi-Head Attention sublayer: split Q, K, V into h heads, run attention in parallel, concatenate, project
2. Add & Norm: residual connection followed by LayerNorm — `LayerNorm(x + MHA(x))`
3. Feed-Forward Network sublayer: two-layer FFN applied position-wise
4. Add & Norm again: `LayerNorm(x + FFN(x))`
5. The block order is: MHA → residual + LayerNorm → FFN → residual + LayerNorm

Key structure: `x = LayerNorm(x + MHA(x))` then `x = LayerNorm(x + FFN(x))`

## Interview Oral Template

> "A standard Transformer encoder block has two sublayers. First, multi-head
> self-attention computes attention across all positions. We add a residual
> connection from the input and apply LayerNorm — this stabilizes training and
> helps gradient flow. Then a position-wise feed-forward network (typically
> two linear layers with ReLU) processes each position independently. Again
> we add a residual connection and LayerNorm. The residual connections are
> critical — without them, deep Transformers can't train effectively."

## Common Pitfalls

- **Wrong normalization order**: Original Transformer uses Post-LN (norm after residual add); many modern models use Pre-LN (norm before sublayer). Know which variant is asked for
- **Forgetting residual connections**: Without residuals, gradients vanish in deep stacks
- **FFN applied globally instead of per-position**: The FFN is the same for all positions but applied independently to each
- **Mismatching dimensions**: MHA output projection and FFN must preserve d_model so residuals can be added

## Complexity

- Time: O(n^2 · d + n · d · d_ff), where d_ff is typically 4d
- Space: O(n^2) for attention + O(n · d_ff) for FFN intermediates
- Parameter count per block: ~12d^2 (4d^2 for MHA projections + 8d^2 for FFN with d_ff = 4d)
