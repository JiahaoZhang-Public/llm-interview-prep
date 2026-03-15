# P026 实现 cross entropy loss

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 从 logits 和 labels 计算 cross entropy。
- 对 batch 求平均。
- 不能直接调用现成 CrossEntropyLoss。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
