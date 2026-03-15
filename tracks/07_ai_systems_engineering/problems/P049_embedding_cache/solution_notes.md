# P049 Embedding Cache — Solution Notes

## Core Approach

1. Cache computed embeddings keyed by the input text (or its hash)
2. On each embedding request, check the cache first
3. Cache hit: return the stored embedding vector directly
4. Cache miss: compute the embedding via the model, store it in the cache, return it
5. Embeddings are deterministic for a given model and input, so caching is always safe

## Interview Oral Template

> "An embedding cache stores previously computed embeddings to avoid redundant
> model inference. Since embeddings are deterministic for a given model and
> input text, caching is always valid -- there's no staleness issue like with
> generative outputs. I hash the input text to create a cache key and store
> the embedding vector. This is especially valuable in RAG pipelines where
> the same documents are re-embedded across queries, or when the same query
> is asked repeatedly. The cache can be in-memory for speed or persisted
> to disk for durability across restarts."

## Common Pitfalls

- **Model version mismatch**: If the embedding model is updated, all cached embeddings become invalid -- include model version in the cache key
- **Memory for large caches**: Each embedding is typically 768-4096 floats (3-16 KB) -- millions of embeddings require significant memory; consider disk-backed storage
- **Batch vs single caching**: When embedding a batch, check each item against the cache individually -- only compute embeddings for cache misses
- **Hash function choice**: Use a fast hash for the input text; the text itself can be long, so hashing is cheaper than string comparison

## Complexity

- Time: O(1) for cache lookup; O(L * d) for a cache miss (embedding computation on text of length L)
- Space: O(N * d) where N is number of cached items and d is embedding dimension
- Cache hit eliminates the most expensive operation (model inference), reducing latency from ~10ms to ~0.01ms
