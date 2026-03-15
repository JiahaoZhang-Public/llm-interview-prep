# P007 实现 causal mask

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 生成 [seq_len, seq_len] 的下三角 mask。
- 屏蔽未来位置。
- 允许指定 mask fill value。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
