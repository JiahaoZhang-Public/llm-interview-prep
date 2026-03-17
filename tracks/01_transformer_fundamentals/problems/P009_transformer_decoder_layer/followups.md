# Interview Follow-ups

## Fundamentals
- How does a decoder layer differ from an encoder layer?
- In an encoder-decoder Transformer, the decoder has 3 sublayers. What are they?
- Why is the causal mask essential for autoregressive generation?

## KV Cache Integration
- How does KV Cache work with a decoder layer during generation?
- During inference with KV Cache, only the new token's Q is needed. Why?
- What changes are needed in the decoder layer's forward pass to support KV Cache?

## Architecture Choices
- GPT uses Post-LN, LLaMA uses Pre-LN with RMSNorm. What are the trade-offs?
- How does parallel attention (computing attention and FFN in parallel) work? Which models use it?
- What is the impact of the number of layers vs. model width on quality?
