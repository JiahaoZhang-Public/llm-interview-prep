# Interview Follow-ups

## Fundamental Understanding
- Why must you accumulate from the highest probability token? What happens if reversed?
- How much practical difference is there between p=0.9 and p=0.95? Why does OpenAI default to p=1?
- If all tokens have equal probability (uniform distribution), what is the nucleus size?

## Comparative Analysis
- What problem of Top-K does Top-P solve? Illustrate with a concrete scenario.
- When combining Top-K + Top-P, what is the execution order? Why?
- Top-P + Temperature: if low T makes the distribution very sharp, the nucleus has only 1 token — is this identical to greedy?

## Engineering & Papers
- What is the core finding of the nucleus sampling paper (Holtzman et al., 2020)?
- How does the `top_p` parameter work in HuggingFace `generate()`? Its relationship with `do_sample`?
- How to efficiently implement nucleus sampling on GPU? (cumsum → binary search for threshold position)
