# P039 Reranker — Solution Notes

## Core Approach

1. Take the initial retrieval results (from embedding similarity search) and re-score them with a more powerful model
2. For each (query, document) pair, compute a relevance score using a cross-encoder or scoring function
3. Sort documents by the new scores in descending order
4. Return the top-k reranked results
5. Two-stage pipeline: fast retrieval (bi-encoder) then accurate reranking (cross-encoder)

## Interview Oral Template

> "A reranker is the second stage in a two-stage retrieval pipeline. The first
> stage uses a bi-encoder for fast candidate retrieval -- query and documents
> are embedded independently. The reranker then takes each query-document
> pair and scores them jointly using a cross-encoder, which is more accurate
> because it can model token-level interactions between query and document.
> The tradeoff is cost: cross-encoders are O(n) forward passes versus one
> for bi-encoders, so we only rerank a small candidate set, typically 20-100
> documents."

## Common Pitfalls

- **Reranking too many documents**: Cross-encoder inference is expensive -- reranking 1000 documents defeats the purpose of the two-stage design
- **Score normalization**: Different rerankers output different score ranges -- may need normalization if combining with other signals
- **Not passing full text**: Cross-encoders need the actual text, not just embeddings -- must store original text alongside vectors
- **Ignoring reranker latency**: Reranking adds latency proportional to the number of candidates -- must balance quality vs response time

## Complexity

- Time: O(N * L * d) for N candidate documents, each requiring a cross-encoder forward pass over L tokens
- Space: O(N) for the scores
- The cross-encoder forward pass is the bottleneck -- typically 10-100ms per document
