# P022 Logits Processor (Repetition Penalty) Solution Notes

## Core Approach

1. Iterate over the **deduplicated set** of generated token ids
2. For each seen token, apply penalty to its logit:
   - Positive logit: divide by penalty (decreases)
   - Negative logit: multiply by penalty (more negative = decreases)
3. Net effect: reduces probability of repeated tokens
4. Unseen token logits remain unchanged

## Interview Oral Template

> "Repetition penalty is a logits processor that modifies logits before softmax to reduce generation repetition.
> For previously seen tokens: positive logits are divided by penalty, negative logits are multiplied by penalty —
> both operations decrease the token's final probability.
> 
> penalty = 1.0 has no effect, > 1.0 suppresses repetition, typical range is 1.1-1.5.
> 
> Important distinction from frequency penalty:
> repetition penalty is binary — only cares whether a token appeared, not how many times;
> frequency penalty (used in OpenAI's API) scales linearly with occurrence count.
> 
> Also note the different handling of positive vs negative logits —
> if you uniformly divide or multiply, the effect on negative logits reverses,
> accidentally *increasing* the repeated token's probability."

## Common Pitfalls

- **Different handling for positive vs negative logits**: The #1 mistake. Positive → divide, negative → multiply. Uniform operation reverses the effect for negatives
- **Deduplication**: generated_ids may contain duplicates, but repetition penalty only checks "appeared or not", so use `set(generated_ids)`
- **Penalty range**: < 1.0 actually *encourages* repetition (valid but uncommon); > 2.0 can derail output
- **EOS handling**: If EOS is penalized, the model may be unable to stop generating
- **logit = 0**: Divide or multiply by penalty both give 0 — no special handling needed
- **Immutability**: Return a new list rather than modifying in place

## Complexity

- Time: O(|unique_generated_ids|) iterating over deduplicated id set
- Space: O(V) for the adjusted logits copy
