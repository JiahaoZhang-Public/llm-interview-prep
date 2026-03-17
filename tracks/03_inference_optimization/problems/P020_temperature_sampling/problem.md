# P020 实现 Temperature Sampling

## 背景

**Temperature** 是控制 LLM 生成随机性的最基础超参数。将 logits 除以 T 后再 softmax：

- **T < 1**：分布更尖锐 → 更确定
- **T = 1**：不变
- **T > 1**：分布更平坦 → 更随机
- **T → 0**：退化为 greedy
- **T → ∞**：退化为均匀分布

```
logits = [2.0, 1.0, 0.5]
T=0.5: probs ≈ [0.84, 0.11, 0.04]   ← 更确定
T=1.0: probs ≈ [0.59, 0.24, 0.17]   ← 原始
T=2.0: probs ≈ [0.43, 0.31, 0.26]   ← 更随机
```

## 接口规范

```python
def temperature_sample(logits: list[float], temperature: float = 1.0) -> int:
    """按 temperature 缩放 logits 后随机采样"""
```

## 约束

| 条件 | 说明 |
|------|------|
| temperature | > 0（不要求处理 T=0） |
| 数值稳定性 | softmax 前减去 max |

## 练习建议

1. 实现 logits / T → softmax → sample
2. 验证低 T 趋向 greedy
3. 说明为什么 T 是"除"不是"乘"
