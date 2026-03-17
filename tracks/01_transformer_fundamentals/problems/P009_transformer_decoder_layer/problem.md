# P009 实现 Transformer Decoder Layer

## 背景

**Transformer Decoder Layer** 是 GPT、LLaMA 等 decoder-only LM 的基本单元。
和 Encoder Block (P004) 的区别：**自带 causal mask**，保证自回归性。

```
x: (B, T, hidden_size)
    ↓ 生成 causal mask（上三角=True/被遮挡）
    ↓ Masked Self-Attention + Residual + LayerNorm
    ↓ FFN + Residual + LayerNorm
output: (B, T, hidden_size)
```

**Decoder-only vs Encoder-Decoder：**
- Decoder-only（GPT, LLaMA）：只有 masked self-attention
- Encoder-Decoder（T5, BART）：decoder 还多一层 cross-attention

## 接口规范

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int): ...
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """如果 mask 未提供，自动生成 causal mask"""
```

## 约束

| 条件 | 说明 |
|------|------|
| 默认 mask | mask=None 时自动生成 causal mask |
| 结构 | masked self-attention + FFN，各带 residual + LN |
| 自回归性 | 位置 i 只能 attend 到 0..i |

## 练习建议

1. 在 P004 基础上增加自动 causal mask 生成
2. 验证 shape 和因果性
3. 口头说明和 encoder block 的区别
