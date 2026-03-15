def rerank_documents(query: str, docs, scorer):
    scored = [(doc, scorer(query, doc)) for doc in docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored]
