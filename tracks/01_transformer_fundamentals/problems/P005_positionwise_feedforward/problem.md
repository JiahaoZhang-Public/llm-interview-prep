# P005 实现 position-wise feedforward

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 实现 FFN(x) = max(0, xW1 + b1)W2 + b2。
- 按位置独立计算。
- 保持 batch 和 seq 维度。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
