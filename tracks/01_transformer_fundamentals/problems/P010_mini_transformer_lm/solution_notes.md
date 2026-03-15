# P010 Mini Transformer LM 题解

## 核心思路

1. **Token + Position Embedding**：token embedding 查表 + 可学习的 position embedding
2. **Transformer Block**：causal self-attention → LayerNorm → FFN → LayerNorm（Pre-LN 或 Post-LN 均可）
3. **LM Head**：最后一层 Linear 映射到 vocab_size，输出 logits
4. **Causal Mask**：上三角 mask 保证 autoregressive，位置 i 只能看到 ≤i 的 token

## 面试口述模板

> "一个最小的 Transformer LM 包含三部分：embedding 层、一个 transformer block、和 LM head。
> Embedding 把 token id 和位置分别映射到 hidden_size 维向量后相加。
> Transformer block 里先做 causal self-attention——用上三角 mask 保证自回归——然后 residual + LayerNorm，
> 接着过一个两层 FFN（expand → GELU → project back），再 residual + LayerNorm。
> 最后 LM head 是一个 Linear(hidden_size → vocab_size)，输出每个位置的 next-token logits。"

## 常见坑

- **causal mask 方向反了**：应该 mask 掉未来位置（上三角为 True/被遮挡）
- **position embedding 长度不够**：要么用固定长度（512），要么用 RoPE 等相对位置编码
- **residual connection 漏掉**：attention/FFN 的输入要加回输出上
- **Pre-LN vs Post-LN**：面试时说清楚你用的哪种；Pre-LN 训练更稳定，现代模型更常用

## 复杂度

- 时间：O(n² · d + n · d · d_ff)，attention 是 n² 瓶颈
- 空间：O(n² + n · d)
- 参数：~4d² + 2d · d_ff + vocab_size · d（单层）
