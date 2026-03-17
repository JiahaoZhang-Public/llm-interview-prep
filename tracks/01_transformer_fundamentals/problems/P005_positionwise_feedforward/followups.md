# Interview Follow-ups

## Fundamentals
- Why use a 4× expansion ratio? What is the intuition behind expand-then-compress?
- Why is this equivalent to two 1×1 convolutions? Draw the analogy.
- What happens if you remove the nonlinearity (making FFN a single linear layer)?

## Activation Functions
- Compare ReLU, GELU, and SwiGLU. Why has each successor improved model quality?
- SwiGLU has 3 weight matrices instead of 2. How does this affect parameter count?
- What is the "dead neuron" problem with ReLU, and how do GELU/SwiGLU address it?

## Engineering
- FFN accounts for ~2/3 of parameters in a Transformer layer. Why?
- How does Mixture of Experts (MoE) relate to the FFN layer?
- In memory-constrained settings, how can you reduce FFN size without losing quality?
