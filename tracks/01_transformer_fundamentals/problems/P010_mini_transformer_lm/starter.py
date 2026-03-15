import torch
from torch import nn


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(512, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))

        return self.lm_head(x)
