# P029 LoRA Forward — Solution Notes

## Core Approach

1. Freeze the original weight matrix W (out × in)
2. Add two low-rank matrices: A (rank × in) and B (out × rank), where rank << min(in, out)
3. Forward pass: `output = x @ W^T + x @ A^T @ B^T × (alpha/rank)`
4. B is initialized to zero → LoRA contribution is zero at training start, preserving pretrained behavior

## Interview Oral Template

> "LoRA constrains weight updates to a low-rank subspace.
> Full fine-tuning learns a ΔW of size (out × in), which is O(d²) parameters.
> LoRA factorizes ΔW into B × A, where A is (rank × in) and B is (out × rank),
> reducing parameters to O(rank × d). Rank is typically 4-64, much smaller than d.
> The forward pass computes W·x + (B·A)·x × scaling, where scaling = alpha/rank.
> B is initialized to zero so training starts without perturbing pretrained representations.
> At inference time, B×A can be merged back into W for zero additional latency."

## Common Pitfalls

- **B not initialized to zero**: Random initialization introduces noise at training start, disrupting pretrained representations
- **Wrong scaling formula**: It's alpha/rank, not just alpha or 1/rank
- **Forgetting to freeze original weights**: LoRA's premise is that W stays fixed; otherwise it's full fine-tuning with extra parameters
- **Rank selection**: Too small = insufficient expressiveness; too large = loses the low-rank advantage. rank=8-16 works well in practice

## Complexity

- Extra forward compute: O(B × rank × in + B × out × rank) = O(B × rank × (in + out))
- Trainable parameters: rank × (in + out), much less than full in × out
- Inference merge: LoRA weights fold into W for zero runtime overhead
