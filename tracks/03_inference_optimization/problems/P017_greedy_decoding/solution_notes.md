# P017 Greedy Decoding Solution Notes

## Core Approach

1. Take argmax of logits to return the highest-probability token id
2. Simplest decoding strategy, fully deterministic, single pass
3. **No softmax needed**: softmax is a monotonically increasing transform that doesn't change argmax

## Interview Oral Template

> "Greedy decoding selects the token with the highest logit at each step — simply argmax.
> No softmax is needed because softmax is monotonic and doesn't change the max position.
> Pros: simple, fast, deterministic. Cons: greedy per-step ≠ globally optimal —
> the best token now might lead to worse choices later.
> Another issue is repetition degeneration: greedy tends to produce 'I think that I think that...' loops.
> That's why real systems use beam search (keeping multiple paths)
> or sampling (top-k/top-p + temperature for randomness)."

## Common Pitfalls

- **Confusing with softmax then argmax**: Greedy only needs argmax, no probability computation
- **No diversity**: Same input always produces same output — unsuitable for creative generation
- **Repetition degeneration**: Greedy is extremely prone to producing repetitive text
- **Tie handling**: When multiple logits are equal, Python's max returns the first — mention this in interviews
- **Batch support**: Real systems use `torch.argmax(logits, dim=-1)` for batched greedy

## Complexity

- Time: O(V), V = vocab size (single pass)
- Space: O(1)
- On GPU, `torch.argmax` is effectively O(V/parallelism), very fast
