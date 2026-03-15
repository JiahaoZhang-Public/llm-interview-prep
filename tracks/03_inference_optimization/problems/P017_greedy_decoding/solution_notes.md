# P017 Greedy Decoding — Solution Notes

## Core Approach

1. At each step, select the token with the highest probability (argmax over logits)
2. Append the selected token to the sequence and feed it back as input for the next step
3. Stop when an EOS token is generated or max_length is reached
4. No randomness — the output is fully deterministic given the same input

Algorithm: `next_token = argmax(logits)` at every step.

## Interview Oral Template

> "Greedy decoding is the simplest generation strategy. At each time step,
> I take the argmax of the output logits to pick the most probable next token.
> I append it to the sequence and repeat until EOS or max length. It's fast
> and deterministic but often produces repetitive or suboptimal text because
> it never explores alternatives. A locally optimal choice at each step
> doesn't guarantee a globally optimal sequence — that's why beam search
> and sampling methods exist."

## Common Pitfalls

- **Repetition loops**: Greedy decoding is prone to degenerate repetition (e.g., "the the the...") — no built-in diversity
- **Not stopping at EOS**: Must check for the EOS token at each step; without this, generation runs to max_length
- **Confusing logits and probabilities**: Argmax of logits equals argmax of softmax(logits), so applying softmax is unnecessary for greedy
- **KV cache**: In practice, greedy decoding should use KV cache to avoid recomputing attention for all previous tokens

## Complexity

- Time: O(T · n · d) where T is generated length, n is total sequence length, d is model dimension
- Space: O(n · d) for KV cache (or O(T · n · d) without cache)
- Fastest decoding method — single forward pass per token with no branching
