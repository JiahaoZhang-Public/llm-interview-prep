import math


def cosine_similarity_search(query_embedding, doc_embeddings, top_k: int = 3):
    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    scores = []
    for doc_id, doc_emb in doc_embeddings.items():
        sim = cosine_sim(query_embedding, doc_emb)
        scores.append((doc_id, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
