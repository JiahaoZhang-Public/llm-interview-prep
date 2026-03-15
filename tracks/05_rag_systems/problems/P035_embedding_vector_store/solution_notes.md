# P035 Embedding Vector Store — Solution Notes

## Core Approach

1. Store document chunks alongside their embedding vectors in memory
2. Add operation: compute embedding for a document chunk and append to the store
3. Search operation: compute query embedding, find top-k most similar document embeddings via cosine similarity
4. Return the corresponding document texts for the top-k matches
5. This is a simplified in-memory version of what FAISS, Pinecone, or Weaviate provide

## Interview Oral Template

> "An embedding vector store maps documents to dense vectors and enables
> similarity search. When adding a document, I compute its embedding and
> store both the vector and the original text. For retrieval, I embed the
> query, compute cosine similarity against all stored vectors, and return
> the top-k most similar documents. This brute-force approach works for
> small collections. For production scale, you'd use approximate nearest
> neighbor indices like FAISS with IVF or HNSW to avoid the linear scan."

## Common Pitfalls

- **Not normalizing vectors**: Cosine similarity requires normalized vectors, or use the formula with norms explicitly; dot product on unnormalized vectors gives wrong rankings
- **Embedding model mismatch**: Query and document embeddings must come from the same model -- mixing models gives meaningless similarity scores
- **Linear scan at scale**: Brute-force O(n) search becomes slow beyond ~100k documents; need ANN indices for production
- **Batch embedding**: Embed documents in batches for efficiency rather than one at a time

## Complexity

- Time: O(n * d) per query for brute-force search (n documents, d dimensions)
- Space: O(n * d) for storing all embeddings
- With ANN indices: query time drops to O(log n) or O(sqrt(n)) depending on the algorithm
