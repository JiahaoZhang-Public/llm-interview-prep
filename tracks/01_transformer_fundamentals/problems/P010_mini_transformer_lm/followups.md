# Interview Follow-ups

## Fundamentals
- What is weight tying between the embedding layer and lm_head? Why does it work?
- How does the model apply position encoding? Compare learned vs. RoPE.
- What loss function is used for language modeling? How is it applied to the logits?

## Training
- Estimate the total parameter count for a model with vocab=32000, d=4096, d_ff=11008, heads=32, layers=32.
- What are the FLOPs for a forward pass? How does this scale with sequence length?
- How does mixed-precision (FP16/BF16) training work for a Transformer LM?

## Architecture Evolution
- Compare the architectures of GPT-2, LLaMA, and Mistral. What are the key differences?
- What is the role of the final LayerNorm before lm_head?
- How would you modify this model to support encoder-decoder or prefix-LM objectives?
