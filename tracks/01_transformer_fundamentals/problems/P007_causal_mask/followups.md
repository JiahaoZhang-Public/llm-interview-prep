# 面试追问

## 基础理解
- mask 应该用 bool 还是 additive logits？PyTorch 的 nn.MultiheadAttention 用的哪种？
- 为什么用 -inf 而不是一个很大的负数（如 -1e9）？有区别吗？
- causal mask 和 padding mask 可以同时用吗？如何组合？

## 扩展
- 如何扩展到多头维度 (B, H, T, T)？需要 unsqueeze 哪些维度？
- 什么是 prefix-LM mask？它和 causal mask 的区别？
- Sliding window attention 的 mask 长什么样？

## 工程问题
- Causal mask 需要每次 forward 重新生成吗？可以缓存吗？
- 在 FlashAttention 中 causal mask 如何高效实现？（不需要 O(n²) 显存）
- 如何在 batch 中处理不同长度的序列？mask 怎么适配？
