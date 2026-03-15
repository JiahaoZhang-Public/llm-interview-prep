# P005 Position-wise Feedforward 题解

## 核心思路

1. 两层线性变换，中间夹激活函数：FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
2. "Position-wise" 意味着对每个位置独立应用同一个 FFN（参数共享跨位置，不共享跨层）
3. 中间层维度 intermediate_size 通常是 hidden_size 的 4 倍

## 面试口述模板

> "Position-wise FFN 是 Transformer 中 attention 之外的另一个关键组件。
> 它是两层线性变换加一个激活函数：先从 hidden_size 扩展到 intermediate_size，
> 经过 ReLU 或 GELU 激活，再投影回 hidden_size。
> 'Position-wise' 是指每个 token 位置独立计算，共享同一组参数。
> 这相当于一个 1x1 卷积，给模型提供了非线性变换的能力。"

## 常见坑

- **激活函数选择**：原始论文用 ReLU，现代模型常用 GELU 或 SwiGLU
- **intermediate_size 比例**：通常 4×hidden_size，但 SwiGLU 结构因为多了一路门控通常用 8/3×
- **忘记输出维度要回到 hidden_size**：中间可以膨胀，但最终输出必须和输入同维

## 复杂度

- 时间：O(n · d · d_ff)，n 为序列长度
- 参数：2 × d × d_ff
