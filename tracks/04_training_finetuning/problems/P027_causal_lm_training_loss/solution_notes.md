# P027 Causal LM Training Loss — Solution Notes

## Core Approach

1. The model predicts the next token at each position: logits at position i predict token at position i+1
2. Shift logits left by 1: `shift_logits = logits[:, :-1, :]`
3. Shift labels right by 1: `shift_labels = labels[:, 1:]`
4. Compute cross-entropy between shifted logits and shifted labels
5. This is the standard next-token prediction / autoregressive language modeling objective

Key operation: align predictions with targets by shifting.

## Interview Oral Template

> "For causal language model training, the loss is next-token prediction via
> cross-entropy. The key implementation detail is the shift — logits at
> position i predict the token at position i+1, so I slice logits as
> positions 0 to T-2 and labels as positions 1 to T-1. Then I reshape
> logits to (batch * seq_len, vocab_size) and labels to (batch * seq_len)
> and pass them to F.cross_entropy. Padding tokens should be excluded
> using ignore_index so they don't contribute to the loss."

## Common Pitfalls

- **Forgetting the shift**: Without shifting, position i's logits are compared to position i's label — the model learns the identity function, not next-token prediction
- **Off-by-one error**: logits[:, :-1] and labels[:, 1:] must have the same length — double check the slicing
- **Not ignoring padding**: Padding tokens in labels should use `ignore_index=-100` (PyTorch convention) so they don't affect the loss
- **Reshape for cross_entropy**: Must flatten to 2D logits and 1D labels — F.cross_entropy doesn't accept 3D logits directly

## Complexity

- Time: O(batch · seq_len · vocab_size) for the cross-entropy computation
- Space: O(batch · seq_len · vocab_size) for logits (the largest tensor in typical LLM training)
- The loss computation itself is cheap compared to the model forward pass
