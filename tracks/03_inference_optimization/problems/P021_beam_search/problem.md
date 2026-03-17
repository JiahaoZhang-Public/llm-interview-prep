# P021 实现 Beam Search

## 背景

**Beam search** 在 greedy（1 条路径）和穷举（所有路径）之间取折中。每步保留 beam_size 个累计得分最高的候选序列。

**关键特性：** beam_size=1 退化为 greedy；使用 log probability 累加避免下溢；确定性算法。

## 接口规范

```python
def beam_search(step_fn, start_tokens, beam_size: int, max_new_tokens: int, eos_token_id=None) -> list:
    """返回得分最高的完整序列"""
```

step_fn(tokens) → [(log_prob, token_id), ...]

## 约束

| 条件 | 说明 |
|------|------|
| beam_size | >= 1 |
| EOS 处理 | 已结束的 beam 不再扩展但参与排序 |
| 全部 EOS | 提前停止 |
| 返回值 | 最高分序列（含 start_tokens） |

## 练习建议

1. 先实现不带 EOS 的版本
2. 加 EOS early stopping
3. 思考 length normalization 的必要性
