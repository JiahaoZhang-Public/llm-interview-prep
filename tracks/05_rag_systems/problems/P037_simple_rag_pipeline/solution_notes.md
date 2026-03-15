# P037 Simple RAG Pipeline — Solution Notes

## Core Approach

1. **Retrieve**: Use the retriever to perform semantic search on the query, fetch top-k relevant documents
2. **Augment**: Concatenate retrieved document texts into a context string, fill into the prompt template
3. **Generate**: Send the augmented prompt to the LLM to produce the final answer

RAG = Retrieval-Augmented Generation — lets the LLM answer based on external knowledge.

## Interview Oral Template

> "A RAG pipeline has three stages: Retrieve, Augment, Generate.
> First, a retriever (typically vector-based) performs semantic search on the user query
> to fetch the top-k most relevant documents.
> Then we concatenate the retrieved texts into a context string and plug it into a prompt template
> alongside the query. Finally, the augmented prompt goes to the LLM for answer generation.
> RAG avoids baking all knowledge into model weights — the knowledge base can be updated independently,
> it reduces hallucination, and it enables source attribution (which documents the answer is based on)."

## Common Pitfalls

- **Context overflow**: Too many or too long documents may exceed the LLM's context window; needs chunking and length control
- **Poor retrieval quality**: Garbage in, garbage out. Embedding model and chunking strategy directly impact end results
- **Prompt template format**: The arrangement and separators between query and context affect LLM comprehension
- **Conflicting documents**: When retrieved documents contradict each other, the LLM may produce inconsistent answers

## Complexity

- Retrieval: O(N) brute-force / O(log N) with vector indices (HNSW, IVF, etc.)
- Generation: Depends on LLM and prompt length
- End-to-end latency = retrieval latency + LLM generation latency (retrieval typically <100ms; generation is the bottleneck)
