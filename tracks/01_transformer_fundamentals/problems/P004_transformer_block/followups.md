# Interview Follow-ups

## Fundamentals
- What is the difference between Pre-LN and Post-LN? Why does Pre-LN train more stably?
- Why are residual connections essential? What happens without them in a 100-layer model?
- Draw the complete data flow through a Transformer block.

## Components & Parameters
- Where would you place dropout in a Transformer block?
- Calculate the total parameter count for a block with d_model=4096, d_ff=11008, num_heads=32.
- Estimate the FLOPs for a forward pass through one block.

## Advanced Topics
- What is gradient checkpointing? How does it trade compute for memory in deep Transformers?
- How does Mixture of Experts (MoE) modify the FFN part of a Transformer block?
- Explain DeepNorm and its approach to stabilizing very deep Transformers.
