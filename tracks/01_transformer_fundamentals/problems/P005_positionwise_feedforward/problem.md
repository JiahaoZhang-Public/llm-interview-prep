# P005 实现 Position-wise Feedforward

## 背景

**Position-wise FFN** 是 Transformer Block 中 attention 之外的另一个关键组件。

公式：`FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂`

```
x: (B, T, d)
    ↓ W₁: (d, d_ff)      # 扩展
    ↓ activation (ReLU/GELU)
    ↓ W₂: (d_ff, d)      # 投影回原维度
output: (B, T, d)
```

**"Position-wise" 含义：** 每个位置（token）独立应用同一组参数，等价于 kernel_size=1 的卷积。
**中间层膨胀比：** 通常 d_ff = 4 × d（如 hidden_size=768, intermediate=3072）。

**激活函数演变：** 原始论文用 ReLU，GPT-2 起用 GELU，LLaMA 用 SwiGLU。

## 接口规范

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int): ...
    def forward(self, x: Tensor) -> Tensor:
```

## 约束

| 条件 | 说明 |
|------|------|
| 两层线性变换 | fc1: hidden→intermediate, fc2: intermediate→hidden |
| 激活函数 | ReLU（本题）或 GELU |
| 输出维度 | 和输入相同 |

## 练习建议

1. 实现 fc1 → relu → fc2
2. 验证 shape：输入输出 hidden_size 不变
3. 口头说明 SwiGLU 和 ReLU 的区别
