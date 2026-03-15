class RoundTripTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def fit(self, corpus):
        token_id = 0
        for text in corpus:
            for word in text.split():
                if word not in self.token_to_id:
                    self.token_to_id[word] = token_id
                    self.id_to_token[token_id] = word
                    token_id += 1

    def encode(self, text: str):
        return [self.token_to_id[word] for word in text.split()]

    def decode(self, token_ids):
        return " ".join(self.id_to_token[tid] for tid in token_ids)
