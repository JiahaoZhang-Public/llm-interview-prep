# P021 Beam Search — Solution Notes

## Core Approach

1. Maintain beam_size candidate sequences (beams), each with a cumulative log-probability
2. At each step, expand every beam with all possible next tokens, producing beam_size × vocab candidates
3. Keep only the top beam_size candidates by cumulative score
4. Stop when all beams hit EOS or max_new_tokens is reached; return the highest-scoring sequence

## Interview Oral Template

> "Beam search is a trade-off between greedy decoding and exhaustive search.
> It maintains beam_size best candidate sequences at each step, rather than just the single best (greedy).
> Each step expands every beam with all possible next tokens, computes cumulative log probabilities,
> and keeps the top B candidates. beam_size=1 degenerates to greedy; larger values approach exhaustive
> search but cost more compute. Note that beam search is deterministic, unlike sampling methods."

## Common Pitfalls

- **Using probability instead of log probability**: Multiplying probabilities causes underflow; always add log probabilities
- **EOS handling**: Finished sequences should not be expanded further but must remain in the candidate pool for ranking
- **Length bias**: Without length normalization, beam search favors shorter sequences (smaller absolute log prob)
- **Larger beam ≠ always better**: Very large beam sizes can produce overly generic outputs

## Complexity

- Time: O(T × B × V), T = generation length, B = beam_size, V = candidates per step
- Space: O(B × T) to store beam sequences
- In practice, step_fn usually returns only the top few candidates, not the full vocab
