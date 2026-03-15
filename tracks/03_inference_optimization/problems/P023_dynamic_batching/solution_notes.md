# P023 Dynamic Batching 题解

## 核心思路

1. 维护一个请求队列
2. enqueue 把新请求加入队列尾部
3. flush 取出最多 max_batch_size 个请求组成 batch
4. 保持 FIFO 顺序

## 面试口述模板

> "Dynamic batching 是 LLM serving 的核心优化之一。
> 不同于静态 batch（等到凑满 N 个请求才处理），
> dynamic batching 根据当前队列状态灵活组 batch。
> 实现上就是一个带上限的 FIFO 队列：请求进来 enqueue，
> 推理引擎定期 flush 取一批处理。
> 更高级的实现还会考虑 continuous batching——
> 已完成的请求可以随时退出 batch，新请求可以随时加入。"

## 常见坑

- **flush 时队列可能为空**：要处理空 batch 的情况
- **请求长度差异**：同一 batch 内不同请求长度不一，需要 padding
- **continuous batching**：flush 不应该阻塞——vLLM 等系统用 iteration-level scheduling

## 复杂度

- enqueue：O(1)
- flush：O(B)，B = batch size
