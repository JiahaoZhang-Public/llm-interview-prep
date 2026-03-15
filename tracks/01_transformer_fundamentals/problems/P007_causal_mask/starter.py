import torch


def causal_mask(seq_len: int, fill_value: float = float('-inf')):
    mask = torch.full((seq_len, seq_len), fill_value)
    mask = torch.triu(mask, diagonal=1)
    return mask
