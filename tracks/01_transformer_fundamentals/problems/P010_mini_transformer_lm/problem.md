# P010 从零实现 Mini Transformer Language Model

## 背景

这道题把前面所有组件**串联**起来，构建一个最小但完整的 Transformer Language Model。

```
input_ids: (B, T)
    ↓ Token Embedding: Embedding(vocab_size, hidden_size)
    ↓ Position Embedding: Embedding(max_seq_len, hidden_size)
    ↓ x = token_emb + pos_emb
    ↓ Transformer Block (with causal mask):
    │     Masked Self-Attention → Residual + LN → FFN → Residual + LN
    ↓ LM Head: Linear(hidden_size, vocab_size)
logits: (B, T, vocab_size)
```

**LM Head 输出 logits**（不做 softmax），配合 cross-entropy loss 训练。
**Weight tying**：LM head 和 token embedding 可共享权重（减少参数）。

## 接口规范

```python
class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, intermediate_size: int): ...
    def forward(self, input_ids: Tensor) -> Tensor:
        """输入 (B, T) token ids，输出 (B, T, vocab_size) logits"""
```

## 约束

| 条件 | 说明 |
|------|------|
| Embedding | token + position |
| 至少一层 | 包含 attention + FFN |
| Causal mask | 自回归 mask |
| LM Head | Linear(hidden, vocab)，输出 raw logits |
| 输出 shape | (B, T, vocab_size) |

## 练习建议

1. 组装所有组件，确保 shape 链路通
2. 用小参数（vocab=32, hidden=16）做 smoke test
3. 口头说明完整的 forward 流程
