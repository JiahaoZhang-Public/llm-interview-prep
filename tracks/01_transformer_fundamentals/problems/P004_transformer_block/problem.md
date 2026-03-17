# P004 实现 Transformer Block

## 背景

**Transformer Block** 是 Transformer 的基本构建块，堆叠 N 个就是完整的 encoder/decoder。

每个 block 包含两个子层，每个子层都有 **residual connection + LayerNorm**：

```
x ──────────────┐
↓               │
Self-Attention   │
↓               │
x = LN(x + attn)┘ ← 残差 + 归一化

x ──────────────┐
↓               │
FFN              │
↓               │
x = LN(x + ffn) ┘ ← 残差 + 归一化
```

**Pre-LN vs Post-LN：**
- Post-LN（原始论文）：`LN(x + Sublayer(x))`
- Pre-LN（现代常用）：`x + Sublayer(LN(x))`
- Pre-LN 训练更稳定，现代 LLM（GPT-3, LLaMA）基本都用 Pre-LN

## 接口规范

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int): ...
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """输入输出 shape 均为 (B, T, hidden_size)"""
```

## 约束

| 条件 | 说明 |
|------|------|
| 子层 | Self-Attention + FFN |
| 残差连接 | 每个子层都有 |
| LayerNorm | 每个子层都有 |
| FFN 激活函数 | GELU（现代标准） |

## 练习建议

1. 先搭框架：attention → residual → LN → FFN → residual → LN
2. 确认输入输出 shape 一致
3. 口头说明 Pre-LN vs Post-LN 的区别和选择理由
