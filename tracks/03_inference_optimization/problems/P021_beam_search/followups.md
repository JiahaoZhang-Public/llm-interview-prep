# 面试追问

## 基础理解
- 为什么要用 log probability？用 probability 会怎样？
- beam_size=1 和 greedy 完全等价吗？
- Beam search 保证全局最优吗？

## 高级变体
- 什么是 length normalization？为什么不加会偏向短序列？
- 什么是 diverse beam search？
- 如何实现 constrained beam search？

## 对比与选择
- 翻译为什么常用 beam search？对话为什么用 sampling？
- Beam search + sampling 的混合策略？
- ChatGPT 为什么不用 beam search？

## 工程实现
- 如何用 batch 推理并行化多个 beam？
- 内存占用优化？
- 扩展阶段 B*V 个候选如何高效取 top-B？
