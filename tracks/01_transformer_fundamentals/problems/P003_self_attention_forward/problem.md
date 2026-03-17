# P003 实现 Self-Attention Forward

## 背景

**Self-attention** 中 "self" 的含义：Q、K、V 都来自**同一个输入** x。

和 P001 的区别：P001 的 Q/K/V 是直接给定的；P003 需要从输入 x **投影**出 Q/K/V。
和 P002 的区别：P003 是单头版本，不做多头拆分。

```
x: (B, T, hidden_size)
    ↓ Q = x @ W_Q   K = x @ W_K   V = x @ W_V    # 三个独立 Linear
    ↓ scores = Q @ K^T / √d
    ↓ if mask: scores += mask
    ↓ weights = softmax(scores)
    ↓ output = weights @ V → (B, T, hidden_size)
```

Self-attention 使每个位置都能和序列中所有位置交互，是 Transformer 捕获长距离依赖的核心。

## 接口规范

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        """三个线性投影 W_Q, W_K, W_V"""

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """输入输出 shape 均为 (B, T, hidden_size)"""
```

## 约束

| 条件 | 说明 |
|------|------|
| Q/K/V | 各自独立的 nn.Linear，不共享参数 |
| scaling | 除以 √hidden_size |
| 输出 | 和输入同 shape |

## 练习建议

1. 实现三个投影 + 标准 attention
2. 思考和 P001、P002 的关系
3. 口头说明 self-attention vs cross-attention 的区别
