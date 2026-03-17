# P018 实现 Top-K Sampling

## 背景

**Top-K sampling** 是一种受限随机采样策略：只保留概率最高的 K 个 token，在 top-K 候选集内重新归一化后随机采样。

**K=1 退化为 greedy；K 的局限性是固定不变**——分布尖锐时 K=50 太多，分布平坦时 K=50 太少。

## 题目目标

实现纯 Python 的 top-k sampling。

## 接口规范

```python
def top_k_sample(logits: list[float], k: int) -> int:
    """在 logits 的 top-k 候选集中随机采样"""
```

## 约束

| 条件 | 说明 |
|------|------|
| k 范围 | 1 <= k <= len(logits) |
| 返回值 | 必须是 top-k 中某个原始 index |
| 数值稳定性 | softmax 前减去 max logit |
| 随机性 | 使用 `random` 模块 |

## 练习建议

1. 排序 → 截取 top-k → softmax → 采样
2. 注意数值稳定性（减 max trick）
3. 口头比较 top-k 和 top-p
