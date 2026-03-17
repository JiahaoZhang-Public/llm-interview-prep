# P023 Dynamic Batching 题解

## 核心思路

1. FIFO 请求队列
2. enqueue 加入尾部
3. flush 取出最多 max_batch_size 个请求
4. 保持插入顺序

## 面试口述模板

> "Dynamic batching 是 LLM serving 基石。简单版是带上限的 FIFO 队列。
> 比 static batching（凑满 N 个才处理）好在不会让请求等太久。
> 但更高级的是 continuous batching：每个 decode step 都可以让完成的请求退出、新请求加入。
> vLLM 和 TGI 用这种策略，吞吐提升 2-3 倍。
> 实际还要考虑：长度差异导致的 padding 浪费、优先级调度、队列超时。"

## 常见坑

- **空队列 flush**：返回空列表
- **请求长度差异**：batch 内需要 padding
- **线程安全**：enqueue 和 flush 在不同线程需要锁
- **flush 时机**：定时 vs 凑够 N 个 → 延迟 vs 吞吐 tradeoff

## 复杂度

- enqueue：O(1)
- flush：O(B)
