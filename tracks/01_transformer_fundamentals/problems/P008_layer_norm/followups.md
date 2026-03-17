# 面试追问

## 基础理解
- 为什么 ε 要放在方差上（分母里的根号下）而不是归一化结果上？
- 为什么用 `unbiased=False`（总体方差）？用 `unbiased=True` 会怎样？
- LayerNorm 的归一化维度为什么是最后一维？对哪些维度求 mean/var？

## 和其他 Normalization 对比
- RMSNorm 和 LayerNorm 的区别？为什么 RMSNorm 更快？
- BatchNorm 为什么不适合 Transformer？（提示：batch 维度统计量问题）
- GroupNorm、InstanceNorm 和 LayerNorm 的关系？

## 工程问题
- 手写 LayerNorm 和 torch.nn.LayerNorm 的性能差异有多大？
- fused LayerNorm kernel 如何优化？（提示：减少 memory access）
- 在 mixed precision 训练中 LayerNorm 通常用什么精度？
