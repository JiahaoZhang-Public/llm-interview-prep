# P033 Implement gradient accumulation

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Accumulate gradients across multiple micro-batches.
- Step the optimizer every accumulation_steps.
- Return the number of optimizer steps taken.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
