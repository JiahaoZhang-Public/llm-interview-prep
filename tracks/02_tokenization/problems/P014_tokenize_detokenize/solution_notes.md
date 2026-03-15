# P014 Tokenize and Detokenize — Solution Notes

## Core Approach

1. **Tokenize (encode)**: Split text into words, map each word to its integer ID using a word-to-index dictionary
2. Handle unknown words by mapping to a special `[UNK]` token ID
3. **Detokenize (decode)**: Map integer IDs back to words using an index-to-word dictionary, then join with spaces
4. The two operations must be inverse of each other for in-vocabulary words: `decode(encode(text)) == text`
5. Build both mappings (word2idx and idx2word) from the vocabulary

## Interview Oral Template

> "A word-level tokenizer maintains a bidirectional mapping between words
> and integer IDs. Encoding splits text by whitespace, looks up each word
> in the word-to-index dictionary, and returns a list of IDs — unknown words
> get the UNK token ID. Decoding reverses this by mapping IDs back to words
> and joining with spaces. The key requirement is round-trip consistency:
> for any in-vocabulary text, decode of encode should recover the original."

## Common Pitfalls

- **Not handling UNK consistently**: Must define a fixed UNK token and its ID; forgetting this causes KeyError on unseen words
- **Losing information on decode**: UNK tokens can't be reversed — original word is lost
- **Whitespace normalization**: Multiple spaces, tabs, or newlines should be handled consistently in both directions
- **Special tokens**: Don't forget [PAD], [BOS], [EOS] etc. — they need IDs too but shouldn't appear in normal text output

## Complexity

- Time: O(n) for both encode and decode, where n is the number of words/tokens
- Space: O(|V|) for the two dictionaries (word2idx and idx2word)
- Dictionary lookups are O(1) average with hash maps
