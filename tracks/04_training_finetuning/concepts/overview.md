# Flashcards

## Why do we shift labels in causal LM training?

Each token predicts the next token, so inputs and targets are offset by one position.

## What does gradient clipping protect against?

It prevents unstable updates when gradients spike.

## Why is LoRA parameter-efficient?

It learns low-rank adapters instead of updating the full dense weight.

## When do you need gradient accumulation?

When the desired effective batch size exceeds memory limits.
