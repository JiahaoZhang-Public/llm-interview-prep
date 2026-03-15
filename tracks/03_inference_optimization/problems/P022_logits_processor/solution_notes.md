# P022 Logits Processor (Repetition Penalty) — Solution Notes

## Core Approach

1. Before sampling, inspect previously generated token IDs
2. For each token that has already appeared, divide its logit by the penalty factor (if logit > 0) or multiply (if logit < 0)
3. Penalty > 1 discourages repetition; penalty = 1 means no effect
4. This is applied to the raw logits before temperature scaling and softmax
5. Common in HuggingFace's `generate()` as `repetition_penalty`

Algorithm: For each already-generated token i, `logits[i] /= penalty` if positive, else `logits[i] *= penalty`.

## Interview Oral Template

> "Repetition penalty is a logits processor that reduces the probability
> of tokens that have already been generated. For each previously seen token,
> I divide its logit by the penalty factor if positive, or multiply if negative —
> both operations reduce the effective probability. This helps prevent the
> degenerate repetition loops common in greedy and low-temperature sampling.
> It's applied to raw logits before temperature and softmax, and is one of
> several logits processors that can be chained together in a generation pipeline."

## Common Pitfalls

- **Symmetric penalty for pos/neg logits**: Must divide positive logits but multiply negative logits — both reduce probability
- **Applying to all vocab tokens**: Only penalize tokens that actually appeared in the generated sequence
- **Penalty too high**: Aggressive penalty (e.g., > 2.0) can force the model into incoherent territory by avoiding all natural word repetitions
- **Order of operations**: Apply repetition penalty before temperature scaling and top-k/top-p filtering

## Complexity

- Time: O(T + V) where T is the number of previously generated tokens to check and V is vocab size for the logits
- Space: O(T) to track generated tokens (often a set for O(1) lookup)
- Very lightweight — a simple element-wise operation on a subset of logits
