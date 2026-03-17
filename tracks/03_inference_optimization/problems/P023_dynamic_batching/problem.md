# P023 实现 Dynamic Batching

## 背景

LLM serving 中请求异步到达。**Dynamic batching** 维护请求队列，根据当前状态灵活组 batch。

更高级的 **Continuous Batching**：已完成的请求立即退出，新请求随时加入（vLLM、TGI 的核心策略）。

## 接口规范

```python
class DynamicBatcher:
    def __init__(self, max_batch_size: int): ...
    def enqueue(self, request): """入队"""
    def flush(self) -> list: """取出最多 max_batch_size 个请求，FIFO 顺序"""
```

## 约束

| 条件 | 说明 |
|------|------|
| 空队列 flush | 返回 `[]` |
| FIFO 顺序 | 先入先出 |
| 不满时 | 可以取少于 max_batch_size |

## 练习建议

1. 用 list 或 deque 实现
2. 思考多线程下的同步
3. 口头说明 dynamic batching vs continuous batching
