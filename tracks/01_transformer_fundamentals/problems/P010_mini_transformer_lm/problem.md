# P010 从零实现 mini Transformer language model

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 支持 token embedding。
- 至少包含一层 attention block。
- 通过 LM head 返回词表 logits。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
