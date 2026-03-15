import math


class InMemoryVectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.documents = {}
        self.embeddings = {}

    def add_document(self, doc_id: str, text: str):
        self.documents[doc_id] = text
        self.embeddings[doc_id] = self.embedder(text)

    def search(self, query: str, top_k: int = 3):
        query_emb = self.embedder(query)
        scores = []
        for doc_id, doc_emb in self.embeddings.items():
            sim = self._cosine_sim(query_emb, doc_emb)
            scores.append((doc_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"doc_id": did, "text": self.documents[did], "score": s} for did, s in scores[:top_k]]

    def _cosine_sim(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
