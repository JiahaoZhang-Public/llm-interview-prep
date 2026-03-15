# P004 Transformer Block 题解

## 核心思路

1. **Self-Attention 子层**：multi-head attention + residual connection + LayerNorm
2. **FFN 子层**：两层全连接网络 (expand → activation → project) + residual + LayerNorm
3. 每个子层都是 `LayerNorm(x + Sublayer(x))` 的结构
4. 这是 Transformer 的基本构建块，堆叠 N 个就是完整的 encoder/decoder

## 面试口述模板

> "一个 Transformer Block 由两个子层组成。
> 第一个是 multi-head self-attention 加 residual 和 LayerNorm；
> 第二个是 position-wise FFN 加 residual 和 LayerNorm。
> 这个结构确保了梯度可以通过 residual 直接回传，
> LayerNorm 稳定每一层的分布。
> 堆叠 N 个这样的 block 就构成了完整模型。"

## 常见坑

- **residual 维度不匹配**：FFN 的输出维度必须等于 hidden_size
- **LayerNorm 位置**：Pre-LN（先 norm 再 sublayer）vs Post-LN（先 sublayer 再 norm），面试要说清
- **忘记 residual**：没有 skip connection 的深层 Transformer 几乎不可训练

## 复杂度

- 时间：O(n² · d + n · d · d_ff)
- 参数：~4d² + 2d · d_ff（attention + FFN）
