def train_subword_vocab(corpus, vocab_size: int):
    char_freq = {}
    for text in corpus:
        for ch in text:
            if ch != " ":
                char_freq[ch] = char_freq.get(ch, 0) + 1

    vocab = {}
    token_id = 0
    for ch in sorted(char_freq.keys()):
        vocab[ch] = token_id
        token_id += 1

    words = []
    for text in corpus:
        for word in text.split():
            words.append(list(word))

    while len(vocab) < vocab_size:
        pairs = {}
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merged = best[0] + best[1]
        vocab[merged] = token_id
        token_id += 1

        new_words = []
        for word in words:
            new_word = list(word)
            i = 0
            while i < len(new_word) - 1:
                if (new_word[i], new_word[i + 1]) == best:
                    new_word[i:i + 2] = [merged]
                else:
                    i += 1
            new_words.append(new_word)
        words = new_words

    return vocab
