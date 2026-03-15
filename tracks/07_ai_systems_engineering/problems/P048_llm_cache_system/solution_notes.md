# P048 LLM Cache System — Solution Notes

## Core Approach

1. Hash the prompt (or prompt + generation parameters) to create a cache key
2. On each request, check if the key exists in the cache
3. Cache hit: return the stored response immediately without calling the LLM
4. Cache miss: call the LLM, store the response keyed by the hash, then return it
5. Use an eviction policy (LRU, TTL, or both) to bound cache size

## Interview Oral Template

> "An LLM cache avoids redundant inference by hashing prompts and storing
> responses. When a request comes in, I hash the full prompt along with
> generation parameters like temperature and max_tokens to create a cache
> key. If the key exists, I return the cached response instantly. On a miss,
> I call the LLM, store the result, and return it. This is especially
> effective for applications with repeated queries -- like FAQ bots or
> template-based prompts. I use LRU eviction to keep the cache within a
> memory budget."

## Common Pitfalls

- **Including non-deterministic params in key**: If temperature > 0, the same prompt can produce different outputs -- caching may return stale results; consider only caching deterministic requests
- **Hash collisions**: Use a strong hash function (SHA-256) -- collisions return wrong responses
- **Cache invalidation**: If the model is updated, all cached responses become stale -- need a versioning mechanism
- **Memory management**: LLM responses can be large -- set a max cache size and use LRU or TTL eviction

## Complexity

- Time: O(L) to hash a prompt of length L; O(1) for cache lookup and storage (with hash map)
- Space: O(cache_size * avg_response_length) for the cache
- Cache hit rate depends on query distribution -- high for repetitive workloads, low for unique queries
