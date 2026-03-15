def pad_sequences(sequences, pad_token_id: int = 0):
    max_len = max(len(seq) for seq in sequences)
    padded = []
    masks = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(list(seq) + [pad_token_id] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return padded, masks
