class EmbeddingCache:
    def __init__(self, embedder):
        raise NotImplementedError("Initialize this class in the starter.")

    def get_embedding(self, text: str):
        raise NotImplementedError("Implement P049.get_embedding().")
