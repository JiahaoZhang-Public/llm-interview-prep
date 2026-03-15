# P046 Model Parallel Scheduler 题解

## 核心思路

1. Pipeline Parallelism 将模型按层切分到多个 GPU（stage），数据按 micro-batch 流水线执行
2. 调度器分配每个 time step 中每个 stage 处理哪个 micro-batch
3. 朴素方案（GPipe）：先所有 micro-batch 前向，再所有反向，存在 bubble（GPU 空闲）
4. 1F1B 方案：前向和反向交替执行，减少 bubble 和峰值内存
5. 输出调度表：schedule[time_step][stage] = (micro_batch_id, "F"/"B")

## 面试口述模板

> "Pipeline 并行调度的核心问题是减少 GPU bubble。我会实现 1F1B 调度：在 warmup 阶段逐步填满流水线，每个 stage 依次开始前向传播。进入稳态后每个 stage 交替做一次前向和一次反向。最后 cooldown 阶段完成剩余反向传播。关键约束是：stage i 的前向必须等 stage i-1 的前向完成，反向必须等 stage i+1 的反向完成。bubble 比例约为 (p-1)/m，p 为 stage 数，m 为 micro-batch 数。"

## 常见坑

- **依赖关系**：前向 stage i 依赖 stage i-1；反向 stage i 依赖 stage i+1，不能违反
- **bubble 计算**：面试常考 bubble 比例公式，要能推导
- **内存峰值**：GPipe 需要保存所有 micro-batch 的激活值，1F1B 只需常数个
- **micro-batch 数量**：m 必须 >= p 才能填满流水线，否则 bubble 严重

## 复杂度

- 时间：O(p + m) 个 time step 完成所有前向和反向，总计算量不变
- 空间：GPipe O(m) 激活缓存，1F1B O(p) 激活缓存
