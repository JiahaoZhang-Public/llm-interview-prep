# P032 SFT Training Loop — Solution Notes

## Core Approach

1. Standard PyTorch training loop for supervised fine-tuning
2. Each iteration: `optimizer.zero_grad()` → `outputs = model(inputs)` → `loss = criterion(outputs, labels)` → `loss.backward()` → `optimizer.step()`
3. Loop over batches from a DataLoader for each epoch
4. Optionally log loss, evaluate on validation set, and save checkpoints
5. For LLM SFT, the loss is causal LM cross-entropy on the response tokens only

## Interview Oral Template

> "The SFT training loop follows the standard PyTorch pattern. For each batch:
> zero the gradients, run the forward pass to get logits, compute cross-entropy
> loss against the target tokens, call backward to compute gradients, then
> optimizer.step to update weights. For instruction-tuning LLMs, I typically
> mask the loss on the prompt tokens so only the response part contributes
> to learning. I also add gradient clipping before the optimizer step and
> log the loss periodically for monitoring."

## Common Pitfalls

- **Forgetting zero_grad**: Without zeroing gradients, they accumulate across batches — this is intentional in gradient accumulation but a bug otherwise
- **Wrong order of operations**: Must be zero_grad → forward → loss → backward → (clip) → step, exactly in this order
- **Not masking prompt tokens**: For instruction-tuning, the loss should only apply to response tokens, not the instruction/prompt
- **Not calling model.train()**: Forgetting `model.train()` means dropout and BatchNorm behave as in eval mode

## Complexity

- Time: O(epochs · batches · (forward + backward)) — backward is typically 2-3x the cost of forward
- Space: O(batch_size · seq_len · vocab_size) for logits + O(P) for gradients and optimizer states
- Memory is the bottleneck — use gradient checkpointing or mixed precision for large models
