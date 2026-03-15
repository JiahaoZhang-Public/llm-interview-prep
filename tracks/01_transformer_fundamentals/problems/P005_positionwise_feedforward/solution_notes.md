# P005 Position-wise Feed-Forward Network — Solution Notes

## Core Approach

1. Two linear transformations with a ReLU activation in between
2. Formula: `FFN(x) = W2 · ReLU(W1 · x + b1) + b2`
3. W1 expands from d_model to d_ff (typically 4 * d_model), W2 projects back down
4. Applied identically and independently to each position in the sequence
5. "Position-wise" means there is no interaction between positions — it's like a 1x1 convolution

## Interview Oral Template

> "The position-wise FFN is a two-layer MLP applied independently to each
> token position. The first linear layer expands the dimension from d_model
> to d_ff — usually 4x — passes through ReLU, and the second layer projects
> back to d_model. It's called position-wise because the same weights are
> shared across all positions but each position is processed independently.
> This is where most of the parameters in a Transformer block live — roughly
> 2/3 of the per-block parameters."

## Common Pitfalls

- **Applying across positions**: The FFN must not mix information across sequence positions — it operates on each position vector independently
- **Wrong expansion ratio**: Standard is d_ff = 4 * d_model; using the wrong ratio changes parameter count significantly
- **Missing bias terms**: Some implementations omit biases for efficiency, but the standard formulation includes them
- **Activation function choice**: Original paper uses ReLU; modern variants use GELU or SwiGLU — know which is expected

## Complexity

- Time: O(n · d_model · d_ff) — linear in sequence length since positions are independent
- Space: O(n · d_ff) for the intermediate activations
- Parameters: 2 · d_model · d_ff + d_model + d_ff (two weight matrices plus biases)
