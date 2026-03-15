# P030 LoRA Merge — Solution Notes

## Core Approach

1. LoRA adds low-rank adapters: the effective weight is `W_eff = W + B @ A * (alpha / rank)`
2. Merging means folding the adapter into the base weight permanently
3. Compute `delta_W = B @ A * (alpha / rank)` and add it to W
4. After merging, the model behaves identically but without the separate A, B matrices
5. This eliminates inference overhead from LoRA — no extra matrix multiplications needed

Key formula: `W_merged = W_base + B @ A * (alpha / rank)`

## Interview Oral Template

> "LoRA merge folds the low-rank adapter back into the base weight matrix.
> During training, LoRA keeps the base weights frozen and learns two small
> matrices A and B such that the effective weight is W plus B times A scaled
> by alpha over rank. To merge, I simply compute B times A, scale by
> alpha over rank, and add the result to the base weight. After merging,
> inference is identical to using the original architecture with no extra
> parameters or computation. This is typically done before deployment to
> remove the adapter overhead."

## Common Pitfalls

- **Forgetting the scaling factor**: The delta must be multiplied by `alpha / rank` — omitting this changes the magnitude significantly
- **Merging in-place during training**: If you merge and then continue training, the adapter matrices still receive gradients — must reset A and B or freeze them
- **Matrix multiplication order**: It's `B @ A` not `A @ B` — A is (rank, d_in), B is (d_out, rank), so B @ A is (d_out, d_in) matching W's shape
- **Multiple LoRA layers**: Must merge each adapted layer independently; missing one means inconsistent behavior

## Complexity

- Time: O(d_out · rank · d_in) for the matrix multiplication B @ A per layer
- Space: O(d_out · d_in) for the merged weight — same as the original
- One-time cost at deployment; inference is then standard matrix multiplication
