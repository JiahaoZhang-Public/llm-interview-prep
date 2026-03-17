# P002 实现 Multi-Head Attention

## 背景

**Multi-head attention** 让模型同时关注序列中不同位置的**不同表示子空间**。

核心思想：把 hidden_size 拆成 H 个 head，每个 head 维度 d_k = hidden_size / H，
各 head 独立做 attention，最后 concat 并投影。

```
输入 x: (B, T, hidden_size)
    ↓ Q = x @ W_Q,  K = x @ W_K,  V = x @ W_V     # 三组线性投影
    ↓ reshape: (B, T, H, d_k) → transpose: (B, H, T, d_k)
    ↓ 每个 head 独立做 scaled dot-product attention
    ↓ concat: (B, T, hidden_size)
    ↓ output = concat @ W_O
```

**为什么多头？** 单头 attention 的 attention pattern 是对所有维度的平均。
多头允许不同 head 学习不同的 pattern（如一个 head 关注语法、另一个关注语义）。

## 题目目标

实现 MultiHeadAttention 类（nn.Module）。

## 接口规范

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True):
        """hidden_size 必须能被 num_heads 整除"""

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """输入输出 shape 均为 (B, T, hidden_size)"""
```

## 约束

| 条件 | 说明 |
|------|------|
| hidden_size % num_heads | 必须为 0 |
| 投影矩阵 | W_Q, W_K, W_V: (hidden, hidden)，W_O: (hidden, hidden) |
| reshape 顺序 | (B,T,C) → (B,T,H,D) → (B,H,T,D) |
| concat | transpose(1,2).contiguous().view(B,T,C) |

## 练习建议

1. 先实现投影 + reshape，确保 shape 正确
2. 加入 scaled dot-product attention
3. 口头画出 shape 变化的全链路
