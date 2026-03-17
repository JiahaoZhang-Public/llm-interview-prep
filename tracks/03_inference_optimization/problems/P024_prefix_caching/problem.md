# P024 Implement Prefix Caching

## Background

In LLM serving, many requests share the same **system prompt**. For example, every ChatGPT conversation starts with the same system instructions.

Recomputing the system prompt's KV cache for every request is wasteful. **Prefix caching** stores the KV cache for shared prefixes, allowing subsequent requests to reuse it, skipping the prefill phase and significantly reducing **Time To First Token (TTFT)**.

```
Request 1: [SYS_PROMPT | user_question_1]
            ↓ Full prefill computation
            Cache SYS_PROMPT's KV

Request 2: [SYS_PROMPT | user_question_2]
            ↓ Cache hit!
            Start prefill from "user_question_2", skip SYS_PROMPT
```

**Lookup strategy: Longest prefix match.**
If cache has [A,B] and [A,B,C], querying [A,B,C,D] should match [A,B,C] (the longer prefix).

## Objective

Implement a prefix cache supporting add and longest-prefix-match lookup.

## Interface

```python
class PrefixCache:
    def __init__(self):
        """Initialize empty cache"""

    def add_prefix(self, prefix_tokens, value):
        """Store a prefix and its associated value (e.g., KV cache reference)"""

    def lookup(self, prefix_tokens):
        """Find the longest matching prefix. Return its value, or None if no match."""
```

## Examples

```python
cache = PrefixCache()
cache.add_prefix((1, 2), "kv_short")
cache.add_prefix((1, 2, 3), "kv_long")

cache.lookup((1, 2, 3, 4))   # → "kv_long"  (matches [1,2,3])
cache.lookup((1, 2, 5))      # → "kv_short" (matches [1,2])
cache.lookup((1,))            # → None       (no match)
cache.lookup((9, 9, 9))      # → None       (no match)
```

## Constraints

| Condition | Description |
|-----------|-------------|
| Longest match | Return the longest matching prefix's value |
| No match | Return None |
| Exact prefix | `query[:len(cached)]` must exactly equal the cached key |
| Empty query | `lookup(())` may return None |

## Advanced Thinking

- Brute-force lookup is O(N * L), N = cache entries, L = query length
- A **Trie (prefix tree)** optimizes this to O(L)
- Real systems also need **LRU eviction** to control memory usage
- Prefix granularity can be token-level or block-level (e.g., SGLang's RadixAttention)

## Practice Tips

1. Start with dict + brute-force longest match
2. Think about how to optimize with a Trie
3. Verbally explain prefix caching benefits in multi-turn conversation scenarios
