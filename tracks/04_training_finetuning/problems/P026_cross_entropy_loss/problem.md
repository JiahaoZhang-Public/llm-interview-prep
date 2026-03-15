# P026 Implement cross entropy loss

## Goal

Implement the smallest correct version of the target component, with attention to correctness, shapes, and edge cases.

## Requirements

- Compute cross entropy from logits and labels.
- Average over the batch.
- Avoid using torch.nn.CrossEntropyLoss directly.

## Practice Hint

- First make the interface and shapes correct
- Then add edge-case handling
- Finish by explaining complexity and engineering tradeoffs out loud
