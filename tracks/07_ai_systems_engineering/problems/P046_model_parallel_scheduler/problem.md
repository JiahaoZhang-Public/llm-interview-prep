# P046 Implement model-parallel inference scheduling

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Assign requests to model shards.
- Track which shard is responsible for each stage.
- Return an executable stage plan.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
