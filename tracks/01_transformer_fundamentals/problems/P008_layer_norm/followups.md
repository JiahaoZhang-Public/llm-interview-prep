# Interview Follow-ups

## Fundamentals
- Why is ε needed? What goes wrong if ε = 0 and all features have the same value?
- Why must we use unbiased=False (population variance) to match PyTorch's LayerNorm?
- Where exactly is LayerNorm placed in a Transformer block? Compare Pre-LN vs Post-LN.

## Normalization Comparisons
- What is RMSNorm? How does it differ from LayerNorm and why is it used in LLaMA?
- Compare LayerNorm, BatchNorm, GroupNorm, and InstanceNorm — when is each appropriate?
- RMSNorm removes the mean-centering step. Why might this still work well?

## Engineering
- How is LayerNorm fused into a single CUDA kernel in practice?
- What is the memory overhead of storing mean/variance for the backward pass?
- In mixed-precision training, LayerNorm is often kept in FP32. Why?
