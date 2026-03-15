class EmbeddingCache:
    def __init__(self, embedder):
        self.embedder = embedder
        self.cache = {}

    def get_embedding(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.embedder(text)
        return self.cache[text]
