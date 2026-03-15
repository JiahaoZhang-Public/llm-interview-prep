# P008 Layer Normalization 题解

## 核心思路

1. 对输入 x 的最后一个维度计算均值 μ 和方差 σ²
2. 归一化：x̂ = (x - μ) / √(σ² + ε)
3. 仿射变换：y = γ · x̂ + β（γ 和 β 是可学习参数）
4. ε 是数值稳定性的小常数，防止除零

## 面试口述模板

> "Layer Normalization 对每个样本的最后一个维度（即 hidden_size）做归一化，
> 先减均值除标准差让分布标准化，然后用可学习的 γ 和 β 做仿射变换恢复表达能力。
> 和 Batch Norm 的区别是：LN 在 feature 维度归一化，不依赖 batch 统计量，
> 所以推理时行为和训练时完全一致，也不受 batch size 影响。
> 这是 Transformer 能稳定训练的重要组件。"

## 常见坑

- **方差计算用 unbiased=False**：LN 用总体方差而非样本方差（分母是 N 不是 N-1）
- **归一化维度搞错**：必须在最后一个维度（hidden_size），不是 batch 或 seq 维度
- **ε 太大或太小**：通常用 1e-5，太小可能数值不稳定，太大会影响精度
- **和 RMSNorm 的区别**：RMSNorm 只除 RMS（不减均值），计算更快，现代 LLM 常用

## 复杂度

- 时间：O(n · d)，n 为序列长度，d 为 hidden_size
- 参数：2d（γ 和 β）
