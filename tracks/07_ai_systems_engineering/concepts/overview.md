# Flashcards

## What is the tradeoff in model-parallel scheduling?

Higher utilization vs. more communication and placement complexity.

## Why rate limit by tokens instead of requests?

Request cost varies widely with prompt and generation lengths.

## Why cache prompts and embeddings separately?

They have different invalidation rules, cost profiles, and hit patterns.

## What should a minimal LLM service API expose?

A stable generation endpoint with clear input, config, and output shape.
