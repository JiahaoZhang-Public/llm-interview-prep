# P031 Parameter Freezing — Solution Notes

## Core Approach

1. Freeze all parameters: set `param.requires_grad = False` for every parameter
2. Selectively unfreeze target parameters by name pattern matching
3. Common patterns: unfreeze only the last layer, classifier head, or adapter layers
4. Use `model.named_parameters()` to iterate and filter by name
5. Only unfrozen parameters receive gradient updates during training

## Interview Oral Template

> "Parameter freezing is used in transfer learning and fine-tuning to limit
> which parts of the model are updated. I first freeze everything by setting
> requires_grad to False for all parameters. Then I iterate through
> named_parameters and selectively unfreeze the ones matching my target
> pattern -- for example, only the final classification layer or LoRA adapter
> weights. This dramatically reduces memory usage because frozen parameters
> don't need gradient storage, and it prevents catastrophic forgetting of
> pretrained knowledge in the frozen layers."

## Common Pitfalls

- **Forgetting to freeze first**: If you only unfreeze targets without freezing everything else first, all parameters remain trainable
- **Optimizer parameter groups**: The optimizer must be created after freezing -- otherwise it tracks all parameters and wastes memory on frozen ones
- **BatchNorm layers**: Freezing BN weights doesn't freeze running statistics -- call `model.eval()` on BN layers or use `requires_grad_` carefully
- **Name pattern errors**: Parameter names depend on module hierarchy -- print `named_parameters()` first to verify the naming scheme

## Complexity

- Time: O(P_unfrozen) for gradient computation during backward -- proportional to unfrozen parameter count
- Space: Saves approximately 2x memory per frozen parameter (no gradient + no optimizer state)
- For LoRA-style fine-tuning, typically less than 1% of parameters are unfrozen
