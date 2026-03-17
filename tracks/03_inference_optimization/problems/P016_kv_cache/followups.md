# Interview Follow-ups

## Fundamental Understanding
- Why is autoregressive generation O(n^3) without KV Cache? How does it become O(n^2) with it?
- KV Cache only stores K and V. Why not cache Q?
- What is the KV Cache shape for multi-head attention? Draw out the dimensions.

## Engineering Implementation
- What's the performance issue with repeated `torch.cat`? How to optimize with pre-allocated buffers?
- Does each Transformer layer need its own KV Cache? Estimate KV Cache memory for LLaMA-70B at 4096 seq_len.
- How to reset after a request ends? How to manage separate caches for concurrent requests?

## Advanced Topics
- What is Paged Attention? How does it solve KV Cache memory fragmentation?
- How does GQA (Grouped Query Attention) affect KV Cache size?
- What is Sliding Window Attention? How does it impact KV Cache?
- Explain how prefix caching reuses KV Cache across different requests.
