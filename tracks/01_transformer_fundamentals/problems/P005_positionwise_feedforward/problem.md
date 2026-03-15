# P005 Implement a position-wise feedforward layer

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Implement FFN(x) = max(0, xW1 + b1)W2 + b2.
- Process each position independently.
- Preserve batch and sequence dimensions.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
