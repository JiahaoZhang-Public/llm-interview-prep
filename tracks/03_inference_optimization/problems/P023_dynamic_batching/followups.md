# 面试追问

## 基础理解
- Dynamic batching 的核心目标？延迟和吞吐如何 tradeoff？
- batch 内不同请求长度不同对 GPU 计算有什么影响？
- flush 的触发时机？

## Continuous Batching
- 什么是 continuous batching / iteration-level scheduling？
- 和 static batching 相比吞吐提升多少？
- 如何处理 batch 内不同请求处于不同生成阶段？

## 工程问题
- 不同线程 enqueue/flush 需要加锁吗？
- 如何实现优先级队列？
- 请求超时如何处理？
- vLLM 调度器如何实现？PagedAttention 如何配合？

## 系统设计
- 1000 QPS 的 LLM serving 系统 batching 策略？
- 如何根据 input length 和 max_new_tokens 调整 batch size？
- padding 浪费如何减少？
