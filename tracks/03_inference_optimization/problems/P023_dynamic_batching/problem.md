# P023 Implement dynamic batching

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Merge concurrent generation requests.
- Support enqueue and flush.
- Preserve request ordering in the emitted batch.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
