# P022 实现 Repetition Penalty Logits Processor

## 背景

**Repetition penalty** 修改 logits 以降低已出现 token 被再次选中的概率。

**公式（Keskar et al., 2019）：**
- logit > 0：`logit / penalty`（变小）
- logit < 0：`logit * penalty`（更负）

两种操作效果一致：降低重复 token 的概率。

## 接口规范

```python
def apply_repetition_penalty(logits: list[float], generated_ids: list[int], penalty: float) -> list[float]:
    """对已生成 token 施加 repetition penalty，返回调整后的 logits"""
```

## 约束

| 条件 | 说明 |
|------|------|
| penalty | >= 1.0（1.0 无效果） |
| generated_ids | 可能为空/有重复（去重处理） |
| 不修改原列表 | 返回新列表 |

## 练习建议

1. 遍历 generated_ids，分正负处理
2. 注意去重
3. 思考 frequency penalty 和 repetition penalty 的区别
