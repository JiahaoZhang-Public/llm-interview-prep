# P003 Self-Attention Forward — Solution Notes

## Core Approach

1. Input x serves as the source for Q, K, and V (hence "self"-attention)
2. Project x through three independent linear layers: Q = W_Q·x, K = W_K·x, V = W_V·x
3. Perform scaled dot-product attention: softmax(QK^T / sqrt(d_k)) · V
4. Key difference from P001: here we create Q, K, V from a single input rather than receiving them pre-computed

Key formula: `SelfAttn(x) = softmax((xW_Q)(xW_K)^T / √d_k) · (xW_V)`

## Interview Oral Template

> "Self-attention means Q, K, and V all come from the same input x.
> We apply three separate linear projections — W_Q, W_K, W_V — to map x
> into query, key, and value spaces, then run standard scaled dot-product
> attention. This lets every position attend to every other position in the
> sequence, which is how Transformers capture long-range dependencies.
> The three projections must be independent — sharing weights would collapse
> the representational capacity."

## Common Pitfalls

- **Confusing self-attention with cross-attention**: In self-attention Q/K/V come from the same input; in cross-attention K/V come from a different sequence (e.g., encoder output)
- **Sharing projection weights**: W_Q, W_K, W_V must be separate nn.Linear modules with independent parameters
- **Forgetting to scale**: Same as P001 — must divide by sqrt(d_k) before softmax
- **Wrong output dimension**: The output shape should match the input shape (batch, seq_len, d_model)

## Complexity

- Time: O(n^2 · d), where n is sequence length, d is model/head dimension
- Space: O(n^2) for attention weights + 3d^2 parameters for the projection matrices
