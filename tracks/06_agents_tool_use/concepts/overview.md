# 知识卡片

## tool calling 的最小闭环是什么？

模型决策工具 -> 工具执行 -> 结果回注给模型。

## 为什么 function schema 必须严格解析？

解析宽松会让工具调用脆弱且容易静默失败。

## short-term memory 应该保存什么？

只保存会影响下一步决策的局部任务状态。

## ReAct 为什么容易调试？

Thought、Action、Observation 三段显式可观测。
