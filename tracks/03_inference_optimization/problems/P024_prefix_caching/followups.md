# 面试追问

## 基础理解
- 为什么 prefix caching 能降低 TTFT？能降低 TPOT 吗？
- Token 前缀如何做 hash？
- 缓存 hit 后从哪个位置继续 prefill？

## 缓存管理
- Prefix cache 何时失效？
- 如何控制内存增长？LRU vs LFU？
- 如何避免 cache stampede？

## 高级实现
- Radix Attention（SGLang）和简单 prefix caching 的区别？
- Block-level prefix caching 如何工作？
- Trie 实现的优劣？

## 系统设计
- 多轮对话场景命中率能有多高？
- 跨 GPU worker 共享 prefix cache 的架构？
- prefix caching 和 Anthropic 的 prompt caching 是同一概念吗？
