# P028 Gradient Clipping — Solution Notes

## Core Approach

1. After `loss.backward()`, compute the global L2 norm of all parameter gradients
2. If the total norm exceeds `max_norm`, scale all gradients down proportionally
3. Use `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` — it returns the original norm
4. Apply clipping after backward but before `optimizer.step()`
5. This prevents exploding gradients from destabilizing training

Formula: if `||g|| > max_norm`, then `g = g * max_norm / ||g||`

## Interview Oral Template

> "Gradient clipping prevents exploding gradients by capping the global L2
> norm of all gradients. After backward pass, I compute the total norm across
> all parameters. If it exceeds max_norm, I scale every gradient down by
> the ratio max_norm over total_norm, preserving the direction but limiting
> the magnitude. This is essential for training Transformers and RNNs where
> gradients can spike. I use clip_grad_norm_ which does this in-place and
> returns the original norm — useful for monitoring training stability."

## Common Pitfalls

- **Clipping before backward**: Clipping must happen after `loss.backward()` and before `optimizer.step()` — clipping before backward is meaningless
- **Per-parameter vs global norm**: `clip_grad_norm_` computes a single global norm across all parameters and scales uniformly; `clip_grad_value_` clips each element independently — they behave very differently
- **Choosing max_norm**: Typical values are 1.0 or 0.5 for LLMs; too large defeats the purpose, too small slows learning
- **Forgetting the underscore**: `clip_grad_norm_` (with trailing underscore) modifies gradients in-place; the non-underscore version is deprecated

## Complexity

- Time: O(P) where P is total number of parameters — one pass to compute norm, one to scale
- Space: O(1) additional — only stores the scalar norm
- Negligible overhead compared to the backward pass itself
