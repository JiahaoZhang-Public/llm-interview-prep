# P025 Streaming Generation — Solution Notes

## Core Approach

1. Use a Python generator function that yields tokens one at a time as they are generated
2. Each iteration: run the model forward pass, select the next token, yield it
3. The caller receives tokens incrementally without waiting for the full sequence
4. Use `yield` to produce each token, maintaining generation state between calls
5. Stop yielding when EOS is produced or max_length is reached

## Interview Oral Template

> "Streaming generation uses a Python generator to yield tokens one by one
> as they're produced, rather than waiting for the entire sequence to finish.
> The function runs the model forward pass, picks the next token via greedy
> or sampling, yields it to the caller, then continues. This is essential
> for chat interfaces where users expect to see text appear progressively.
> Under the hood, the generator preserves its local state between yields,
> so the KV cache and running sequence are maintained naturally."

## Common Pitfalls

- **Not using yield**: Using a list and appending defeats the purpose — must use `yield` for true streaming
- **Forgetting KV cache**: Without caching, each yield requires reprocessing the entire sequence from scratch
- **EOS handling**: Must stop the generator when EOS is produced — don't yield the EOS token itself to the user
- **Backpressure**: If the consumer is slower than generation, tokens queue up in memory — consider async generators for production use

## Complexity

- Time: O(d) per token with KV cache (single token forward pass), O(T · n · d) total for T tokens
- Space: O(n · layers · d) for the KV cache that persists across yields
- Latency: Time-to-first-token equals one forward pass; subsequent tokens arrive at model speed
