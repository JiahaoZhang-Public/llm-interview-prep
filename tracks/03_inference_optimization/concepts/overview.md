# 知识卡片

## KV cache 主要节省了什么？

避免每个新 token 都重复计算历史位置的 K/V。

## 为什么需要 top-p 而不只用 top-k？

top-p 会随分布形状自适应候选集合大小。

## dynamic batching 解决了什么问题？

把并发请求合并解码，提升吞吐并减少空转。

## prefix caching 对对话场景为什么很有价值？

大量请求共享长前缀，缓存后可省掉重复 prefill 成本。
