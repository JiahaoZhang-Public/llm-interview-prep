# P029 Implement LoRA forward

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Implement W' = W + BA.
- Expose a forward path over the adapted weight.
- Support scaling by alpha / rank.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
