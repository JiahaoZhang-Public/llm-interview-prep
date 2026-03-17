# Interview Follow-ups

## Fundamentals
- What is the difference between a boolean mask and an additive (-inf) mask? When would you use each?
- Why use -inf instead of a very large negative number like -1e9? Are there numerical considerations?
- How does the causal mask interact with softmax to produce zero attention weights?

## Extensions
- How would you modify the causal mask for prefix-LM (bidirectional attention on a prefix, causal on the rest)?
- What is sliding window attention? How does its mask differ from a standard causal mask?
- In encoder-decoder models, where is the causal mask applied?

## Engineering
- FlashAttention handles the causal mask implicitly without materializing the full matrix. How?
- For very long sequences, what is the memory cost of storing the mask? How can you avoid it?
- How do you handle causal masking correctly when using KV Cache during generation?
