# P019 实现 Top-P (Nucleus) Sampling

## 背景

**Top-P sampling**（nucleus sampling，Holtzman et al., 2020）是自适应采样策略。
与 Top-K 不同，Top-P 根据累积概率动态确定候选集大小。

**算法**：softmax → 按概率降序排序 → 累加到 >= p → 候选集即 nucleus → 重新归一化 → 采样

p=1.0 = 全分布采样，p→0 = greedy。分布尖锐时 nucleus 小，分布平坦时 nucleus 大。

## 接口规范

```python
def top_p_sample(logits: list[float], p: float) -> int:
    """在累积概率 >= p 的最小候选集中随机采样"""
```

## 约束

| 条件 | 说明 |
|------|------|
| p 范围 | 0 < p <= 1.0 |
| 排序方向 | 从概率最高开始累加 |
| 归一化 | nucleus 内采样前须重新归一化 |

## 练习建议

1. 分步：softmax → sort → cumsum → 截断 → renormalize → sample
2. 思考 p=1.0 和 p→0 的退化行为
3. 口头对比 top-k 和 top-p 的核心区别
