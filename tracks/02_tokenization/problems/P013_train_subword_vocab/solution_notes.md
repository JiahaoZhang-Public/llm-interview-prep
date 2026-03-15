# P013 Train Subword Vocabulary — Solution Notes

## Core Approach

1. Initialize vocabulary with all individual characters from the training corpus
2. Count frequencies of all adjacent token pairs in the tokenized corpus
3. Merge the most frequent pair into a new token and add it to the vocabulary
4. Re-tokenize the corpus with the new token and repeat
5. Stop when vocabulary reaches the desired size or no more merges are beneficial

This is the BPE (Byte Pair Encoding) training algorithm used by GPT models.

## Interview Oral Template

> "BPE vocabulary training starts with character-level tokens and iteratively
> merges the most frequent adjacent pair. Each merge creates a new token and
> reduces the total token count. I repeat until the vocabulary reaches the
> target size. The merge rules are saved in order — during tokenization,
> I apply them in the same order to encode new text. This bottom-up approach
> naturally creates common subwords like 'ing', 'tion', and frequent whole
> words, giving a good balance between vocabulary size and token granularity."

## Common Pitfalls

- **Not tracking merge order**: The order of merges matters during encoding — must be saved and replayed in sequence
- **Counting across word boundaries**: BPE pairs should only be counted within words, not across word boundaries
- **Inefficient re-counting**: Naive re-counting of all pairs after each merge is O(n) per iteration; use incremental updates for efficiency
- **Off-by-one in vocab size**: Initial character vocab counts toward the target size

## Complexity

- Time: O(V_target · n) where n is corpus length in tokens — each merge step scans the corpus
- Space: O(n) for the tokenized corpus + O(V) for the vocabulary
- Training is offline and one-time; the resulting merge table is compact
