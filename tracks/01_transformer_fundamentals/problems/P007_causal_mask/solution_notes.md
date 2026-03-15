# P007 Causal Mask — Solution Notes

## Core Approach

1. Generate an upper-triangular matrix filled with -inf (or a very large negative number)
2. The diagonal and below are 0 — these are the positions a token is allowed to attend to
3. Mask shape is (seq_len, seq_len); position i can only attend to positions 0..i
4. Add this mask to the attention scores before softmax: scores + mask
5. After softmax, -inf positions become 0 probability — no information leaks from future tokens

Implementation: `mask = torch.triu(torch.ones(n, n) * float('-inf'), diagonal=1)`

## Interview Oral Template

> "The causal mask enforces autoregressive attention — each position can only
> attend to itself and earlier positions. I create an upper-triangular matrix
> of negative infinity and add it to the raw attention scores before softmax.
> The negative infinity values become zero after softmax, so no future
> information leaks through. This is essential for decoder-only language
> models like GPT, where each token must be predicted using only its left
> context."

## Common Pitfalls

- **Using 0/1 multiplicative mask**: Multiplying scores by 0 gives 0 (not -inf), which softmax maps to a nonzero probability — must use additive -inf mask
- **Wrong triangle direction**: `torch.triu` with diagonal=1 masks the upper triangle (future); using `tril` would mask the past
- **Boolean vs float mask**: PyTorch's `masked_fill` expects a boolean mask and a fill value; the additive approach expects a float mask — don't mix them up
- **Forgetting batch/head dimensions**: The mask is (seq, seq) but attention scores are (batch, heads, seq, seq) — broadcasting must align correctly

## Complexity

- Time: O(n^2) to create the mask (one-time cost)
- Space: O(n^2) for the mask matrix
- Can be precomputed and reused across layers and batches
