class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}

    def train(self, corpus):
        words = []
        for text in corpus:
            for word in text.split():
                words.append(list(word) + ["</w>"])

        vocab = {}
        for word in words:
            key = tuple(word)
            vocab[key] = vocab.get(key, 0) + 1

        base_tokens = set()
        for word in vocab:
            for ch in word:
                base_tokens.add(ch)
        token_id = 0
        self.vocab = {}
        for t in sorted(base_tokens):
            self.vocab[t] = token_id
            token_id += 1

        while len(self.vocab) < self.vocab_size:
            pairs = {}
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            merged = best[0] + best[1]
            self.vocab[merged] = token_id
            token_id += 1

            new_vocab = {}
            for word, freq in vocab.items():
                new_word = list(word)
                i = 0
                while i < len(new_word) - 1:
                    if (new_word[i], new_word[i + 1]) == best:
                        new_word[i:i + 2] = [merged]
                    else:
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab

    def encode(self, text: str):
        tokens_list = []
        for word in text.split():
            symbols = list(word) + ["</w>"]
            for a, b in self.merges:
                merged = a + b
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols[i:i + 2] = [merged]
                    else:
                        i += 1
            for s in symbols:
                if s in self.vocab:
                    tokens_list.append(self.vocab[s])
        return tokens_list

    def decode(self, tokens):
        id_to_token = {v: k for k, v in self.vocab.items()}
        parts = [id_to_token.get(t, "") for t in tokens]
        text = "".join(parts)
        text = text.replace("</w>", " ")
        return text.strip()
