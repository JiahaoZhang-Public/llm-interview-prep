# P036 Cosine Similarity Search — Solution Notes

## Core Approach

1. Compute cosine similarity between the query vector and each document vector
2. Formula: `cos_sim(a, b) = (a . b) / (||a|| * ||b||)`
3. If vectors are pre-normalized to unit length, cosine similarity equals the dot product
4. Sort by similarity score in descending order and return top-k results
5. Can be vectorized: `scores = query @ docs.T` for batch computation on normalized vectors

## Interview Oral Template

> "Cosine similarity measures the angle between two vectors, ignoring magnitude.
> I compute the dot product of the query and document vectors divided by
> the product of their norms. A score of 1 means identical direction, 0 means
> orthogonal, and -1 means opposite. For retrieval, I compute similarity
> against all document embeddings and return the top-k highest scores.
> Pre-normalizing all vectors to unit length simplifies this to a single
> matrix multiplication, which is very efficient on GPU."

## Common Pitfalls

- **Zero vector handling**: Division by zero norm produces NaN -- must handle zero vectors as a special case
- **Not pre-normalizing**: Computing norms every query is wasteful -- normalize document vectors once at index time
- **Confusing with Euclidean distance**: Cosine similarity and L2 distance rank differently for unnormalized vectors; for normalized vectors, they give equivalent rankings
- **Numerical precision**: For very high-dimensional vectors, floating point accumulation errors can affect results -- use float32 minimum

## Complexity

- Time: O(n * d) for computing similarity with all n documents, plus O(n log k) for top-k selection
- Space: O(n) for the similarity scores
- With pre-normalized vectors, this is a single matrix-vector multiply -- highly parallelizable
