import math


def mmr_select(query_embedding, doc_embeddings, lambda_mult: float = 0.5, top_k: int = 3):
    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    selected = []
    remaining = list(range(len(doc_embeddings)))

    while len(selected) < top_k and remaining:
        best_idx = None
        best_score = float('-inf')
        for idx in remaining:
            relevance = cosine_sim(query_embedding, doc_embeddings[idx])
            redundancy = 0.0
            for sel_idx in selected:
                sim = cosine_sim(doc_embeddings[idx], doc_embeddings[sel_idx])
                redundancy = max(redundancy, sim)
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected
