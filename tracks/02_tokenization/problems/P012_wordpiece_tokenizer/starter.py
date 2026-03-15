class WordPieceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token = "[UNK]"

    def encode(self, text: str):
        tokens = []
        for word in text.split():
            sub_tokens = self._tokenize_word(word)
            tokens.extend(sub_tokens)
        return tokens

    def _tokenize_word(self, word):
        sub_tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    found = substr
                    break
                end -= 1
            if found is None:
                return [self.vocab.get(self.unk_token, 0)]
            sub_tokens.append(self.vocab[found])
            start = end
        return sub_tokens

    def decode(self, tokens):
        id_to_token = {v: k for k, v in self.vocab.items()}
        parts = []
        for t in tokens:
            token_str = id_to_token.get(t, self.unk_token)
            if token_str.startswith("##"):
                parts.append(token_str[2:])
            else:
                if parts:
                    parts.append(" ")
                parts.append(token_str)
        return "".join(parts)
