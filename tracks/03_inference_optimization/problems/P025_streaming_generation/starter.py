def stream_generate(step_fn, prompt_ids, max_new_tokens: int, eos_token_id=None):
    tokens = list(prompt_ids)
    for _ in range(max_new_tokens):
        next_token = step_fn(tokens)
        tokens.append(next_token)
        yield next_token
        if eos_token_id is not None and next_token == eos_token_id:
            break
