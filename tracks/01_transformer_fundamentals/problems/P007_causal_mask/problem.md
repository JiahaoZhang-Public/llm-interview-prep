# P007 Implement a causal mask

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Generate a lower-triangular mask of shape [seq_len, seq_len].
- Mask future positions.
- Allow selecting the fill value.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
