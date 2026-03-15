# P020 Temperature Sampling — Solution Notes

## Core Approach

1. Divide logits by a temperature parameter T before applying softmax
2. `probs = softmax(logits / T)`
3. Sample from the resulting probability distribution
4. T > 1 flattens the distribution (more random/creative), T < 1 sharpens it (more deterministic)
5. T → 0 approaches greedy decoding; T → inf approaches uniform random sampling

Key formula: `P(token_i) = exp(logit_i / T) / sum(exp(logit_j / T))`

## Interview Oral Template

> "Temperature sampling controls the randomness of generation by scaling
> logits before softmax. Dividing by T greater than 1 flattens the
> distribution — all tokens become more equally likely, producing diverse
> but potentially incoherent text. Dividing by T less than 1 sharpens the
> distribution — the model becomes more confident in its top choices, making
> output more focused. In the limit, T approaching zero is greedy decoding.
> Temperature is often the first knob to tune for generation quality."

## Common Pitfalls

- **Temperature of 0**: Division by zero — must handle T=0 as a special case (use argmax/greedy instead)
- **Applying temperature after softmax**: Must apply before softmax — dividing probabilities by T is mathematically different
- **Confusing with top-k/top-p**: Temperature changes the shape of the full distribution; top-k/top-p truncate it — they're complementary techniques
- **Wrong scaling direction**: Dividing by T > 1 flattens (not sharpens); a common confusion

## Complexity

- Time: O(V) to scale logits and sample, where V is vocabulary size
- Space: O(V) for the probability distribution
- Negligible overhead on top of the model forward pass
