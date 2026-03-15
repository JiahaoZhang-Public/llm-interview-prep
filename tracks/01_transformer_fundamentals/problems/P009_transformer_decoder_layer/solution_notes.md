# P009 Transformer Decoder Layer 题解

## 核心思路

1. 和 Transformer Block (P004) 结构类似，但自带 causal mask
2. Self-attention 使用上三角 mask 保证自回归
3. Residual + LayerNorm + FFN + Residual + LayerNorm
4. 完整的 decoder 还可能包含 cross-attention（编解码模型），但纯 decoder LM 只有 self-attention

## 面试口述模板

> "Transformer Decoder Layer 是 GPT 类模型的基本单元。
> 和 encoder block 的区别是它在 self-attention 中使用 causal mask，
> 保证每个位置只能看到自己和之前的 token。
> 结构是：masked self-attention + residual + LayerNorm → FFN + residual + LayerNorm。
> 纯 decoder 模型（如 GPT）不需要 cross-attention 子层。
> 堆叠 N 个 decoder layer 加上 embedding 和 LM head 就是完整的语言模型。"

## 常见坑

- **漏加 causal mask**：不加 mask 的 decoder 等于 encoder，失去自回归性
- **和 encoder-decoder 的 decoder 混淆**：encoder-decoder 的 decoder 多了一个 cross-attention 子层
- **mask 生成时机**：mask 的大小随序列长度变化，要在 forward 里动态生成
- **Pre-LN 的好处**：先 LayerNorm 再 attention/FFN，训练更稳定

## 复杂度

- 时间：O(n² · d + n · d · d_ff)，同 Transformer Block
- 参数：同 P004
- 推理时配合 KV Cache 每步只算一个 token 的 attention
