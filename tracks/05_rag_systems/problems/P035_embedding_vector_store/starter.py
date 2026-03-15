class InMemoryVectorStore:
    def __init__(self, embedder):
        raise NotImplementedError("Initialize this class in the starter.")

    def add_document(self, doc_id: str, text: str):
        raise NotImplementedError("Implement P035.add_document().")

    def search(self, query: str, top_k: int = 3):
        raise NotImplementedError("Implement P035.search().")
