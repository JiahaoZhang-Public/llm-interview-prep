# Interview Follow-ups

## Fundamentals
- Why does rotating pairs of dimensions encode relative position? Prove that Q_m · K_n depends only on (m-n).
- Why does RoPE only apply to Q and K, not V?
- Compare RoPE with sinusoidal position encoding (original Transformer) and learned position embeddings.

## Length Extrapolation
- What happens when you use RoPE at lengths longer than training? Why does it degrade?
- Explain NTK-aware RoPE scaling. How does it differ from simple linear interpolation?
- What is YaRN? How does it combine NTK scaling with attention temperature adjustment?

## Engineering
- How is RoPE implemented efficiently in production (precomputed sin/cos tables)?
- How does RoPE interact with KV Cache during incremental decoding?
- In ALiBi (Attention with Linear Biases), position is encoded differently. Compare with RoPE.
