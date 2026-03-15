# P011 BPE Tokenizer — Solution Notes

## Core Approach

1. **Initialize**: Split each word in the corpus into characters + end-of-word marker `</w>`, count frequencies
2. **Train (greedy merging)**: Repeatedly find the most frequent adjacent pair across the corpus, merge it into a new token, add to vocab. Repeat until vocab_size is reached
3. **Encode**: For new text, replay merges in training order to merge character pairs, then look up token IDs
4. **Decode**: Map token IDs back to token strings, concatenate, replace `</w>` with spaces

## Interview Oral Template

> "BPE is a bottom-up greedy merge algorithm.
> During training, we split every word into characters plus an end-of-word marker,
> then repeatedly find the highest-frequency adjacent pair across the corpus and merge it
> into a new token. Each merge adds one entry to the vocabulary until we hit the target size.
> At encode time, we replay the merge rules in order on the input text.
> At decode time, we reverse-map IDs to strings.
> BPE handles OOV gracefully by falling back to character-level tokens,
> while common words get merged into single tokens for efficiency."

## Common Pitfalls

- **Merge order dependency**: Encoding must replay merges in the exact training order, otherwise tokenization is inconsistent
- **Forgetting `</w>`**: Without the end-of-word marker, word boundaries are lost and decoding has no spaces
- **Not weighting by word frequency**: When counting pairs, multiply by word frequency — don't count each unique word once
- **vocab_size includes base characters**: Initial character set already uses some IDs; actual merges = vocab_size - initial charset size

## Complexity

- Training: O(V × N × L), V = target vocab_size, N = vocabulary entries, L = average word length
- Encoding: O(M × L²), M = number of merges, L = word length
- In practice, use priority queues or tries for acceleration
