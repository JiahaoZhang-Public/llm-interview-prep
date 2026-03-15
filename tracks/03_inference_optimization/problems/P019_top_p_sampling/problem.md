# P019 Implement top-p sampling

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Use nucleus sampling.
- Build the smallest candidate set whose cumulative probability exceeds p.
- Return one sampled token id.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
