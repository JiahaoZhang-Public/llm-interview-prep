# Interview Follow-ups

## Fundamentals
- Why divide by √d_k rather than d_k or some other value? What happens if you remove the scaling?
- When the softmax input is very large, what happens to the gradients? How does this relate to the scaling?
- What are the different types of masks (causal, padding, cross-attention), and when is each used?

## Variants & Comparisons
- How does cross-attention differ from self-attention in terms of Q, K, V sources?
- How does mask broadcasting work when batch size > 1 or with multiple heads?
- What happens to attention weights when the sequence is very long? Relate to the "attention sink" phenomenon.

## Advanced Topics
- What is FlashAttention? How does it reduce memory from O(n²) to O(n) without materializing the full attention matrix?
- Compare MHA (Multi-Head Attention), MQA (Multi-Query Attention), and GQA (Grouped-Query Attention).
- What is linear attention? How does it approximate softmax attention in O(n) time?
