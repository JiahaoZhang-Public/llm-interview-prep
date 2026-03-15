# P015 Padding and Attention Mask — Solution Notes

## Core Approach

1. Find the maximum sequence length in the batch (or use a specified max_length)
2. Pad all sequences to this length with a PAD token ID (typically 0)
3. Generate an attention mask: 1 for real tokens, 0 for padding positions
4. The attention mask tells the model to ignore padding during attention computation
5. Both padding and mask are needed for batched processing of variable-length sequences

## Interview Oral Template

> "When batching variable-length sequences, we pad shorter ones to the max
> length with a special PAD token. But we need the model to ignore these
> padding positions, so we create an attention mask — a binary tensor with
> 1s for real tokens and 0s for padding. This mask is used in the attention
> computation to prevent padding tokens from contributing to attention scores.
> In practice, the mask is expanded and converted to additive form: 0s stay
> as 0 and padding positions become negative infinity before softmax."

## Common Pitfalls

- **Padding side**: Left-padding vs right-padding matters — GPT-style models often prefer left-padding for generation so the last token is always meaningful
- **Mask shape mismatch**: The attention mask (batch, seq_len) must be correctly broadcast to (batch, heads, seq_q, seq_k) in the attention layer
- **Padding token gradient**: PAD embeddings can still receive gradients unless explicitly handled — some implementations set PAD embedding to zero and freeze it
- **Not returning the mask**: The mask must be passed alongside the padded input to every attention layer in the model

## Complexity

- Time: O(batch_size · max_len) for both padding and mask generation
- Space: O(batch_size · max_len) for the padded tensor and mask tensor
- This is preprocessing overhead — negligible compared to model forward pass
