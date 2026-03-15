# Flashcards

## Why scale the attention scores?

It stabilizes gradients by preventing large dot products from saturating softmax.

## What is the role of the residual path?

It preserves the identity signal and helps optimization through deep stacks.

## Why does RoPE work well for decoding?

It encodes relative position information directly into Q and K without extra embeddings.

## Why is causal masking mandatory in decoder-only LMs?

It prevents leakage from future tokens during both training and inference.
