# P008 Implement LayerNorm without torch.nn.LayerNorm

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Normalize over the last hidden dimension.
- Apply learnable gamma and beta.
- Do not call the built-in LayerNorm module.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
