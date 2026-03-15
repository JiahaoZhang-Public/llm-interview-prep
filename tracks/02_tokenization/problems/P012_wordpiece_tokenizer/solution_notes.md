# P012 WordPiece Tokenizer — Solution Notes

## Core Approach

1. Start with the full word; try to match the longest prefix in the vocabulary
2. If the full word is in vocab, return it as a single token
3. Otherwise, greedily find the longest matching prefix, then continue with the remainder prefixed by "##"
4. The "##" prefix indicates a continuation subword (not the start of a word)
5. If any remaining substring cannot be found in vocab, output `[UNK]`

Algorithm: greedy longest-match from left to right, with "##" for non-initial pieces.

## Interview Oral Template

> "WordPiece tokenization works by greedy longest-match. Given a word, I try
> to find the longest prefix that exists in the vocabulary. If the whole word
> is a token, great — return it. Otherwise, take the longest matching prefix
> as the first token, then continue with the remainder, prepending '##' to
> indicate it's a continuation piece. If at any point no match is found,
> the entire word maps to UNK. This is used in BERT-family models and
> balances vocabulary size with the ability to handle rare words."

## Common Pitfalls

- **Forgetting the ## prefix**: Continuation tokens must start with "##" — without it, the tokenizer can't distinguish word-initial vs mid-word subwords
- **Not handling UNK**: If any substring has no vocab match, the entire word should be UNK, not just the unmatched part
- **Greedy direction**: WordPiece matches longest prefix first (left-to-right greedy), unlike BPE which merges bottom-up by frequency
- **Whitespace handling**: Words must be pre-split by whitespace before applying WordPiece

## Complexity

- Time: O(w^2) per word in worst case (w = word length), trying all prefix lengths
- Space: O(|V|) for the vocabulary lookup (typically a hash set)
- With a trie, prefix lookup can be optimized to O(w) per word
