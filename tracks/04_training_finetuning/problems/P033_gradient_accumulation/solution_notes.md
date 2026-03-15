# P033 Gradient Accumulation — Solution Notes

## Core Approach

1. Instead of updating weights every batch, accumulate gradients over N mini-batches
2. Call `loss.backward()` each step but only call `optimizer.step()` and `optimizer.zero_grad()` every N steps
3. Divide the loss by N (or equivalently scale the learning rate) to keep the effective gradient magnitude correct
4. This simulates a larger batch size without requiring more GPU memory
5. Effective batch size = mini_batch_size * accumulation_steps

## Interview Oral Template

> "Gradient accumulation simulates a larger batch size when GPU memory is
> limited. I run forward and backward on small mini-batches, but only call
> optimizer.step every N steps. Gradients accumulate naturally since I skip
> zero_grad between accumulation steps. I divide the loss by N to normalize —
> otherwise the accumulated gradient is N times too large. This is essential
> for training large models where the desired batch size doesn't fit in memory.
> The result is mathematically equivalent to training with N times the mini-batch size."

## Common Pitfalls

- **Forgetting to scale the loss**: Must divide by N — otherwise the effective learning rate is N times larger, causing instability
- **zero_grad at the wrong time**: Call zero_grad only after optimizer.step, not after every backward — otherwise you lose accumulated gradients
- **Gradient clipping timing**: Clip gradients only at the optimizer.step boundary, not after each mini-batch backward
- **Distributed training interaction**: When using DDP, gradient sync happens at backward — use `no_sync()` context for non-update steps to avoid unnecessary all-reduce

## Complexity

- Time: Same total computation as large batch — no savings in FLOPs
- Space: O(mini_batch_size · ...) for activations — this is the memory saving
- Gradients are O(P) regardless of accumulation steps — they're summed in-place
