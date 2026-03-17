# Interview Follow-ups

## Fundamental Understanding
- Why can prefix caching reduce TTFT? Can it reduce TPOT (time per output token)?
- What are the approaches for hashing token prefixes? What's wrong with using tuples directly as keys?
- After a cache hit, from which position does prefill continue? How does it interface with KV cache?

## Cache Management
- When does a prefix cache entry become stale? What happens if one word in the system prompt changes?
- How to control cache memory growth? Is LRU or LFU more appropriate?
- If multiple requests miss the same prefix simultaneously, how to avoid duplicate computation? (cache stampede)

## Advanced Implementations
- What is Radix Attention (SGLang)? How does it differ from simple prefix caching?
- How does block-level prefix caching (e.g., vLLM's automatic prefix caching) work?
- What are the pros/cons of a Trie implementation? Compare with hash map approach.

## System Design
- How high can prefix cache hit rates be in multi-turn conversation scenarios?
- To share prefix cache across multiple GPU workers, how would you design the architecture?
- Is prefix caching the same concept as prompt caching (e.g., Anthropic's prompt caching feature)?
