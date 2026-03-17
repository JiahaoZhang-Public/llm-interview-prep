# P020 Temperature Sampling Solution Notes

## Core Approach

1. Divide logits by temperature: `scaled_logits = logits / T`
2. Apply softmax to scaled logits to get probabilities
3. Sample randomly according to probabilities
4. T < 1 sharpens (toward greedy), T > 1 flattens (more random), T = 1 no change

**Mathematical intuition:**
- In softmax(z/T), T scales the differences between logits
- Original logit gap = Δ; after dividing by T, gap = Δ/T
- T < 1 → Δ/T > Δ → gap amplified → sharper distribution
- T > 1 → Δ/T < Δ → gap shrunk → flatter distribution

## Interview Oral Template

> "Temperature is the most fundamental knob for controlling LLM generation randomness.
> It works by dividing logits by T before softmax:
> T < 1 amplifies logit differences, making high-probability tokens even more dominant — more deterministic.
> T > 1 shrinks differences, flattening the distribution — more random.
> T → 0 degenerates to greedy, T → ∞ becomes uniform.
> Typical values: 0-0.2 for code generation (accuracy needed), 0.7-1.0 for creative writing.
> Two key implementation notes: T=0 must be special-cased to argmax,
> and low T amplifies logits which may cause exp overflow, so subtract max before softmax."

## Common Pitfalls

- **temperature = 0 division by zero**: Must special-case as greedy (not required in this problem, but mention in interviews)
- **Scale then softmax**: Not softmax then scale — they're mathematically different
- **Numerical stability**: Low temperature amplifies logits, exp may overflow → subtract max(scaled) first
- **Combination order with top-k/top-p**: Standard practice is temperature scale first, then top-k/top-p truncation
- **Default values across APIs**: OpenAI defaults T=1, Anthropic defaults T=1

## Complexity

- Time: O(V) (traversal + softmax + sampling)
- Space: O(V) (storing scaled logits and probs)
