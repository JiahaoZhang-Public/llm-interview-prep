# P024 Prefix Caching Solution Notes

## Core Approach

1. Maintain a prefix → value mapping (key is a token tuple)
2. `add_prefix` stores entries in the cache
3. `lookup` iterates all cached keys to find the **longest prefix match**
4. If multiple prefixes match, return the longest one's value; return None if no match

**Practical application**: The value is typically a KV cache reference; on hit, skip prefill for the cached portion.

## Interview Oral Template

> "Prefix caching exploits the fact that many requests share the same prefix —
> e.g., all requests start with the same system prompt.
> The idea: cache the prefix's KV cache, and on subsequent requests,
> lookup the longest matching cached prefix and reuse its KV cache,
> skipping that portion of the prefill computation.
> 
> My implementation uses a dict mapping prefix→value, with brute-force iteration to find the longest match.
> Complexity is O(N*L), where N is the number of cache entries and L is query length.
> This can be optimized to O(L) using a Trie (prefix tree).
> 
> Real systems (like SGLang's RadixAttention) also need:
> LRU eviction for memory control, block-level granularity for caching, and cross-GPU cache sharing."

## Common Pitfalls

- **Longest match vs exact match**: Must find the longest prefix match, not require exact equality
- **Query shorter than cache**: If cache has [1,2,3] and query is [1,2], there's no match (the cache is not a prefix of the query in the useful sense)
- **Cache invalidation**: System prompt changes invalidate corresponding cache entries
- **Memory management**: Too many cached prefixes consume significant GPU memory; need LRU eviction
- **Cache stampede**: Multiple requests simultaneously missing the same prefix cause redundant computation

## Complexity

- add: O(L) for hashing (L = prefix length)
- lookup (brute force): O(N * L), N = cache entry count, L = query length
- lookup (Trie): O(L), independent of cache entry count
- Space: O(N * L_avg) storing all prefix keys
