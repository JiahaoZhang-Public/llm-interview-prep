# P040 Multi-Query Retrieval — Solution Notes

## Core Approach

1. Generate multiple query variants from the original user query (e.g., rephrasings, sub-questions)
2. Run retrieval for each query variant independently
3. Collect all retrieved documents and deduplicate by document ID or content
4. Optionally score documents by how many queries retrieved them (reciprocal rank fusion)
5. Return the merged, deduplicated result set

## Interview Oral Template

> "Multi-query retrieval improves recall by generating several variants of
> the user's query and retrieving documents for each. A single query might
> miss relevant documents due to vocabulary mismatch, but different
> phrasings catch different results. I generate variants using an LLM or
> simple paraphrasing, run each through the retriever, then merge and
> deduplicate the results. Documents retrieved by multiple variants are
> likely more relevant, so I can rank by retrieval frequency or use
> reciprocal rank fusion to combine scores."

## Common Pitfalls

- **Too many variants**: Each variant costs a retrieval call -- 3-5 variants is typical; more adds latency without proportional recall gains
- **Duplicate results**: Must deduplicate by document ID, not just text -- same document retrieved by different queries should appear once
- **Query quality**: Poor variant generation (e.g., near-identical rephrases) wastes retrieval calls without improving diversity
- **Score combination**: Simply unioning results loses ranking information -- use reciprocal rank fusion or voting to maintain quality ordering

## Complexity

- Time: O(Q * retrieval_cost) where Q is the number of query variants, plus O(Q * k) for deduplication
- Space: O(Q * k) for all retrieved results before deduplication
- The LLM call for generating variants may dominate latency if not cached
