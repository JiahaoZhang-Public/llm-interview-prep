# P008 Layer Normalization — Solution Notes

## Core Approach

1. Compute mean and variance across the last dimension (feature dimension) for each token
2. Normalize: `x_hat = (x - mean) / sqrt(variance + eps)`
3. Apply learnable affine transform: `output = gamma * x_hat + beta`
4. eps (typically 1e-5) prevents division by zero
5. Unlike BatchNorm, LayerNorm normalizes across features (not across the batch), making it independent of batch size

Formula: `LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta`

## Interview Oral Template

> "LayerNorm normalizes each token's feature vector independently. For each
> position, I compute the mean and variance across the feature dimension,
> subtract the mean, divide by the standard deviation plus a small epsilon
> for stability, then apply a learnable scale (gamma) and shift (beta).
> The key advantage over BatchNorm is that it doesn't depend on batch size
> or other samples in the batch, which is critical for variable-length
> sequences and autoregressive generation where batch statistics are unreliable."

## Common Pitfalls

- **Wrong normalization axis**: Must normalize across the last dimension (features), not across the batch or sequence dimension
- **Forgetting epsilon**: Without eps, zero-variance features cause division by zero
- **Confusing with BatchNorm**: BatchNorm normalizes across the batch dimension and uses running statistics at inference; LayerNorm always uses per-sample statistics
- **Gamma/beta initialization**: gamma is initialized to 1 and beta to 0 — so LayerNorm starts as a simple normalization

## Complexity

- Time: O(n · d), where n is sequence length and d is feature dimension
- Space: O(d) for learnable parameters gamma and beta
- Computation is lightweight compared to attention — not a bottleneck
