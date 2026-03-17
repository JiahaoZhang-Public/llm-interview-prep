# P008 不调用现成 API 实现 LayerNorm

## 背景

**Layer Normalization** 是 Transformer 能稳定训练的关键组件。

公式：
```
μ = mean(x, dim=-1)           # 对最后一维求均值
σ² = var(x, dim=-1)           # 对最后一维求方差（总体方差，不是样本方差）
x̂ = (x - μ) / √(σ² + ε)      # 归一化
y = γ · x̂ + β                 # 仿射变换（γ, β 可学习）
```

**和 BatchNorm 的区别：**
- LayerNorm 对每个样本的 feature 维度归一化
- BatchNorm 对 batch 维度归一化
- LN 不依赖 batch 统计量，推理时行为和训练一致

**重要：** 方差用 `unbiased=False`（分母是 N，不是 N-1）。

## 接口规范

```python
def layer_norm(
    x: Tensor,        # (*, hidden_size)
    gamma: Tensor,    # (hidden_size,)
    beta: Tensor,     # (hidden_size,)
    eps: float = 1e-5
) -> Tensor:
```

## 约束

| 条件 | 说明 |
|------|------|
| 归一化维度 | 最后一个维度（hidden_size） |
| 方差 | 总体方差 `unbiased=False` |
| ε | 加在方差上再开方，防止除零 |
| 不用 nn.LayerNorm | 手写实现 |

## 练习建议

1. 分步：mean → var → normalize → affine
2. 验证输出 mean ≈ 0（gamma=1, beta=0 时）
3. 口头说明 LN vs BN vs RMSNorm
