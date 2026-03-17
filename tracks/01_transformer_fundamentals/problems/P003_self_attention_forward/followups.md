# Interview Follow-ups

## Fundamentals
- What is the difference between self-attention and cross-attention?
- Why do we need three separate projection matrices (W_Q, W_K, W_V) instead of using X directly?
- If seq_len = 1, what does self-attention reduce to?

## Component Relations
- How does this single-head self-attention relate to multi-head attention (P002)?
- In an encoder-decoder model, where is self-attention used vs. cross-attention?
- How does adding a causal mask turn self-attention into masked self-attention?

## Advanced Topics
- What is the computational complexity of self-attention? Why is it O(n²d)?
- How do efficient attention methods (Linformer, Performer) reduce this complexity?
- Explain how KV Cache (P016) optimizes self-attention during autoregressive generation.
