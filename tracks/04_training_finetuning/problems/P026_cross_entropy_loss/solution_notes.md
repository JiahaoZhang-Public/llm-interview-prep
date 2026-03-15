# P026 Cross-Entropy Loss — Solution Notes

## Core Approach

1. Given logits (raw model output) and target class labels, compute the cross-entropy loss
2. Use `F.cross_entropy(logits, targets)` which internally applies log-softmax + NLL loss
3. Logits shape: (batch_size, num_classes); targets shape: (batch_size,) with integer class indices
4. The loss encourages the model to assign high probability to the correct class
5. For numerical stability, PyTorch combines log-softmax and NLL in a single fused operation

Formula: `CE = -sum(y_i * log(softmax(z_i)))` = `-log(softmax(z)[target_class])`

## Interview Oral Template

> "Cross-entropy loss measures how well the predicted probability distribution
> matches the true label. For classification, I pass raw logits and integer
> targets to F.cross_entropy, which internally computes log-softmax and
> then negative log-likelihood. This fused operation is numerically more
> stable than computing softmax and log separately, because it avoids
> overflow in the exponentials. The gradient pushes logits of the correct
> class higher and incorrect classes lower."

## Common Pitfalls

- **Applying softmax before F.cross_entropy**: The function expects raw logits — applying softmax first results in double-softmax and wrong gradients
- **Wrong target format**: Targets must be class indices (Long tensor), not one-hot vectors, for `F.cross_entropy`
- **Ignoring padding tokens**: For sequence tasks, set `ignore_index=pad_token_id` to exclude padding from the loss
- **Reduction mode**: Default is `mean` over the batch; use `reduction='none'` to get per-sample losses for weighted training

## Complexity

- Time: O(batch_size · num_classes) for the softmax and loss computation
- Space: O(batch_size · num_classes) for intermediate log-softmax values
- A single backward pass through cross-entropy is also O(batch_size · num_classes)
