# P045 实现 ReAct agent loop

## 题目目标

围绕指定接口实现一个可测试的最小版本，重点是正确性、shape 推导和边界处理。

## 要求

- 实现 Thought / Action / Observation 循环。
- 模型请求工具时执行 tool。
- 在最终答案处停止。

## 练习建议

- 先只把接口和 shape 跑通
- 再补边界条件和异常输入
- 最后口头说明时间复杂度与工程 tradeoff
