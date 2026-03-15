# P018 Top-K Sampling — Solution Notes

## Core Approach

1. Sort logits by value descending, keep only the top k entries
2. Apply softmax to these k logits to get a probability distribution
3. Sample one token from this distribution

Top-K limits the candidate set, preventing low-probability tokens from being sampled and producing garbage output.

## Interview Oral Template

> "Top-K sampling sorts the logits and keeps only the K highest-probability candidate tokens,
> zeroing out everything else. Then we renormalize over those K candidates and sample randomly.
> This prevents long-tail low-probability tokens from producing nonsensical output.
> The downside is that K is fixed — sometimes the distribution is sharp and only 2-3 candidates
> make sense, sometimes it's flat and you need more. That's the problem Top-P (nucleus) sampling solves."

## Common Pitfalls

- **Forgetting to renormalize**: After selecting top-k, probabilities must sum to 1 within the candidate set
- **k=1 degenerates to greedy**: Worth mentioning in interviews to show understanding
- **Logits vs probabilities**: Standard practice is to select top-k first, then apply softmax (not the other way around)
- **Numerical stability**: Subtract max logit before exp to prevent overflow

## Complexity

- Time: O(V log V) for sorting, V is vocab size; or O(V + K log K) with partial sort
- Space: O(V)
