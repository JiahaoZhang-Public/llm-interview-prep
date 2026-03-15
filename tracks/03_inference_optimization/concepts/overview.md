# Flashcards

## What does KV cache save?

It avoids recomputing historical keys and values for every generated token.

## Why use top-p instead of only top-k?

Top-p adapts the candidate set size to the shape of the distribution.

## What problem does dynamic batching solve?

It improves throughput by merging concurrent requests with similar decoding steps.

## Why is prefix caching valuable for chat workloads?

Many requests share a long prompt prefix, so the cache removes duplicated prefill cost.
