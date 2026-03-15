# P011 实现 BPE tokenizer

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 从语料训练 merge pair。
- 每次 merge 后更新词表。
- 提供 encode 和 decode。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
