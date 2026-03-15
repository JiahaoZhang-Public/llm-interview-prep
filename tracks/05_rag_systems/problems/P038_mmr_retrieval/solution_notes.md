# P038 MMR Retrieval — Solution Notes

## Core Approach

1. MMR (Maximal Marginal Relevance) balances relevance and diversity in retrieval results
2. Formula: `MMR = argmax_{d in R\S} [lambda * sim(d, q) - (1-lambda) * max_{d' in S} sim(d, d')]`
3. Greedy iteration: each round, select the document with the highest MMR score and add it to result set S
4. lambda near 1 prioritizes relevance; lambda near 0 maximizes diversity
5. Implementation: first retrieve a larger candidate set (e.g., top-20), then run MMR to select top-k from candidates

## Interview Oral Template

> "MMR is a greedy reranking algorithm that selects documents balancing
> relevance to the query and diversity among selected results. I first
> retrieve a large candidate set, then iteratively pick the document
> with the highest MMR score -- which is lambda times relevance to the query
> minus one-minus-lambda times maximum similarity to any already-selected
> document. The first document is always the most relevant one since the
> penalty term starts at zero. Each subsequent pick considers redundancy,
> so the results cover more diverse aspects of the query."

## Common Pitfalls

- **Candidate set too small**: If candidates equal k, MMR degrades to plain relevance sorting -- need candidates much larger than k
- **Lambda tuning**: Lambda is the key hyperparameter -- interviewers often ask about the relevance-diversity tradeoff
- **First document special case**: When S is empty, the max-similarity penalty is 0, so the first pick is purely by relevance
- **Precompute pairwise similarity**: Computing candidate-to-candidate similarity each iteration is wasteful -- precompute the N x N matrix

## Complexity

- Time: O(k * N * d) for k selection rounds over N candidates with d-dimensional embeddings
- Space: O(N^2) if precomputing pairwise similarity matrix, otherwise O(N * d)
- k is typically small (5-10), so the overhead over basic retrieval is modest
