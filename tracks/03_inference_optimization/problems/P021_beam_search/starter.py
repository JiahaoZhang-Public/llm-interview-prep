import math


def beam_search(step_fn, start_tokens, beam_size: int, max_new_tokens: int, eos_token_id=None):
    beams = [(0.0, list(start_tokens))]

    for _ in range(max_new_tokens):
        all_candidates = []
        for score, seq in beams:
            if eos_token_id is not None and seq[-1] == eos_token_id:
                all_candidates.append((score, seq))
                continue
            next_steps = step_fn(seq)
            for log_prob, token in next_steps:
                new_seq = seq + [token]
                all_candidates.append((score + log_prob, new_seq))

        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_size]

        if eos_token_id is not None and all(seq[-1] == eos_token_id for _, seq in beams):
            break

    best_score, best_seq = max(beams, key=lambda x: x[0])
    return best_seq
