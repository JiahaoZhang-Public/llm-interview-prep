import torch


def apply_rope(q, k, position_ids=None, base: float = 10000.0):
    seq_len = q.size(-2)
    dim = q.size(-1)

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device).float()
    else:
        position_ids = position_ids.float()

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    angles = position_ids.unsqueeze(-1) * freqs.unsqueeze(0)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    def rotate(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos.repeat_interleave(2, dim=-1) + rotated * sin.repeat_interleave(2, dim=-1)

    return rotate(q), rotate(k)
