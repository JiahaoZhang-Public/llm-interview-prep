# P019 实现 top-p sampling

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 实现 nucleus sampling。
- 构造累计概率超过 p 的最小候选集合。
- 返回一个 token id。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
