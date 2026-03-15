# P044 实现 memory system

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 保存最近消息或 observation。
- 支持 append 和 recent retrieval。
- 限制短期记忆窗口大小。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
