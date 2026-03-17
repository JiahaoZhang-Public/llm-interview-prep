# Interview Follow-ups

## Fundamental Understanding
- Why is greedy decoding deterministic? When is this an advantage vs disadvantage?
- Why is softmax unnecessary before argmax? Is argmax invariant under softmax? Why?
- On which tasks does greedy decoding work well? Where does it fail?

## Comparative Analysis
- Greedy vs Beam Search: How does beam search avoid greedy's local optima problem?
- Greedy vs Sampling: In what scenarios is sampling preferable to greedy?
- If beam_size=1, is beam search exactly equivalent to greedy?

## Engineering Questions
- How to correctly stop greedy generation at EOS?
- How to implement a batch version of greedy (processing multiple sequences simultaneously)?
- What is repetition degeneration? Why is greedy particularly susceptible?
