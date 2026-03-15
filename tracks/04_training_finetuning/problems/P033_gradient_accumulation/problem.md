# P033 实现 gradient accumulation

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 跨多个 micro-batch 累积梯度。
- 每 accumulation_steps 次做一次 optimizer step。
- 返回 optimizer step 次数。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
