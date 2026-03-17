# Interview Follow-ups

## Fundamentals
- Walk through the reshape from `(batch, seq, d_model)` to `(batch, num_heads, seq, head_dim)`. What does each `transpose` do?
- Why use multiple heads instead of a single large attention? What does each head learn?
- How do you correctly apply a mask across all heads?

## Variants & Comparisons
- Compare MHA, MQA (Multi-Query Attention), and GQA (Grouped-Query Attention). What are the KV cache savings?
- What is fused QKV projection? Why is it more efficient than three separate projections?
- How does the number of heads affect model quality vs. compute cost?

## Engineering
- How do you parallelize multi-head attention across GPUs (tensor parallelism)?
- What is the memory cost of the attention matrix for long sequences? How does head count affect it?
- How do production frameworks handle variable-length sequences in batched MHA?
