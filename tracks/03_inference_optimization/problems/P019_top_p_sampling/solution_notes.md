# P019 Top-P (Nucleus) Sampling — Solution Notes

## Core Approach

1. Apply softmax to logits to get probabilities
2. Sort by probability descending
3. Accumulate probabilities until the cumulative sum ≥ p; these tokens form the "nucleus"
4. Renormalize within the nucleus and sample

Top-P's candidate set size is dynamic, adapting to the distribution shape.

## Interview Oral Template

> "Top-P sampling, also called nucleus sampling, doesn't fix the number of candidates like Top-K.
> Instead, it sorts tokens by probability from highest to lowest and accumulates until the
> cumulative probability reaches the threshold p (e.g., 0.9). Those tokens become the candidate set.
> When the distribution is sharp, the nucleus is small (near-greedy); when it's flat, it's large
> (preserving diversity). This adapts better than a fixed K.
> p=1.0 means sampling from the full distribution; p approaching 0 means greedy."

## Common Pitfalls

- **Wrong accumulation direction**: Must accumulate from highest probability, not lowest
- **Boundary handling**: The token that pushes cumulative probability ≥ p should be included
- **Renormalization**: Nucleus probabilities may sum to > p; renormalize to 1.0 before sampling
- **Combined with Top-K**: Production systems often use Top-K + Top-P together — Top-K truncates first, then Top-P filters

## Complexity

- Time: O(V log V) for sorting
- Space: O(V)
- The actual nucleus is usually much smaller than V, so sampling is fast
