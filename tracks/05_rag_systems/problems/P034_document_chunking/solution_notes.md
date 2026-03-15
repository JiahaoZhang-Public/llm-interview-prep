# P034 Document Chunking — Solution Notes

## Core Approach

1. Split a long document into smaller chunks of a specified size (in characters or tokens)
2. Use overlapping windows: each chunk shares some content with the previous one
3. Parameters: `chunk_size` (max characters per chunk) and `overlap` (shared characters between consecutive chunks)
4. Stride = chunk_size - overlap; start each new chunk at the previous start + stride
5. Overlap preserves context at chunk boundaries so retrieval doesn't miss relevant passages

## Interview Oral Template

> "Document chunking splits long text into overlapping segments for embedding
> and retrieval. I slide a window of chunk_size characters across the text
> with a stride of chunk_size minus overlap. The overlap ensures that
> information near chunk boundaries isn't lost -- if a relevant sentence
> spans two chunks, the overlap means it appears fully in at least one.
> Typical values are 500-1000 characters per chunk with 100-200 overlap.
> More sophisticated approaches split on sentence or paragraph boundaries
> to avoid cutting mid-sentence."

## Common Pitfalls

- **No overlap**: Without overlap, sentences at chunk boundaries get split and may not be retrievable
- **Overlap too large**: If overlap approaches chunk_size, you get massive redundancy and near-duplicate embeddings
- **Splitting mid-word**: Character-based chunking can cut words -- prefer splitting at word or sentence boundaries
- **Chunk size vs embedding model limits**: Chunks must fit within the embedding model's max token length (e.g., 512 for many models)

## Complexity

- Time: O(n) where n is document length -- single pass with stride
- Space: O(n * chunk_size / stride) for all chunks, slightly more than n due to overlap
- Chunking is a preprocessing step -- negligible compared to embedding computation
