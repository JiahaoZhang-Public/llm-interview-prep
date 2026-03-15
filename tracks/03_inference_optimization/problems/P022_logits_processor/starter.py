def apply_repetition_penalty(logits, generated_ids, penalty: float):
    adjusted = list(logits)
    for token_id in set(generated_ids):
        if token_id < len(adjusted):
            if adjusted[token_id] > 0:
                adjusted[token_id] /= penalty
            else:
                adjusted[token_id] *= penalty
    return adjusted
