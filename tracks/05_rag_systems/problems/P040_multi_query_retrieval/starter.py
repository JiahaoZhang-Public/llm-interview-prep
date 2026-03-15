def multi_query_retrieve(query: str, query_generator, retriever, top_k: int = 3):
    queries = query_generator(query)
    seen_ids = set()
    results = []
    for q in queries:
        docs = retriever.search(q, top_k=top_k)
        for doc in docs:
            if doc["doc_id"] not in seen_ids:
                seen_ids.add(doc["doc_id"])
                results.append(doc)
    return results[:top_k]
