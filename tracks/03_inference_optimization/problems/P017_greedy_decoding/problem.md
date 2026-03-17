# P017 实现 Greedy Decoding

## 背景

**Greedy decoding** 是最简单的文本生成策略：每一步选择概率最高的 token（即 logits 的 argmax）。

虽然简单高效且**完全确定性**，但 greedy 有根本问题：**当前步最优 ≠ 全局最优**。
容易产生**重复退化**（repetition degeneration）。

## 题目目标

实现纯 Python 的 greedy decode 函数。

## 接口规范

```python
def greedy_decode(logits: list[float]) -> int:
    """返回 logits 中最大值的索引"""
```

## 示例

```python
greedy_decode([0.1, 0.9, 0.2])  # → 1
greedy_decode([5.0, 5.0, 1.0])  # → 0（并列时返回第一个）
greedy_decode([-1.0, -2.0])     # → 0
```

## 约束

| 条件 | 说明 |
|------|------|
| logits 长度 | >= 1 |
| 值范围 | 任意浮点，包括负数 |
| 并列处理 | 返回第一个最大值 |
| 不需要 softmax | softmax 是单调变换，不改变 argmax |

## 练习建议

1. 先写最朴素的遍历版本
2. 思考：为什么不需要先做 softmax？
3. 口头对比 greedy vs beam search vs sampling
