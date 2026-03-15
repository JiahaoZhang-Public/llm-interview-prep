# P024 Prefix Caching — Solution Notes

## Core Approach

1. Store computed KV pairs keyed by their token prefix (e.g., system prompt tokens)
2. On a new request, find the longest matching prefix in the cache
3. Reuse cached KV pairs and only compute attention for the new (suffix) tokens
4. This avoids redundant computation when many requests share the same system prompt or context prefix
5. Cache key is typically a tuple or hash of the token ID sequence

## Interview Oral Template

> "Prefix caching stores the KV cache for common token prefixes — like a
> shared system prompt — so we don't recompute them for every request.
> When a new request arrives, I look up the longest matching prefix in the
> cache, load those KV pairs, and only run the forward pass on the new tokens.
> This is especially valuable for chat applications where every turn starts
> with the same system prompt and conversation history. The lookup uses
> longest-prefix matching, similar to a trie or just iterating from longest
> to shortest cached prefix."

## Common Pitfalls

- **Cache invalidation**: If the model weights change (e.g., after fine-tuning), all cached KV pairs are stale and must be cleared
- **Memory management**: KV caches are large (proportional to layers x heads x seq_len x d_head) — need an eviction policy (LRU is common)
- **Hash collisions**: If using hash-based lookup, ensure the hash is collision-resistant for token sequences
- **Partial prefix match**: Must handle the case where the cached prefix is shorter than the request prefix — recompute only the gap

## Complexity

- Time savings: O(L_prefix · d) per request where L_prefix is the shared prefix length
- Space: O(cache_entries · L · layers · d) for stored KV pairs
- Lookup: O(L) for exact prefix matching via hash or trie
