# 面试追问

## 基础理解
- Pre-LN 和 Post-LN 的区别是什么？各自的优缺点？现代模型用哪种？
- 为什么 residual connection 对深层 Transformer 至关重要？
- FFN 中 GELU 和 ReLU 的区别？为什么现代模型更常用 GELU？

## 组件选择
- dropout 通常加在哪些位置？（attention weights、FFN 输出、residual）
- 如何估算一个 Transformer Block 的参数量？给出公式。
- 如何估算一个 Block 的 FLOPs？哪部分计算量最大？

## 高级话题
- 如何做 activation checkpointing 来减少显存？代价是什么？
- Mixture of Experts (MoE) 如何替换 FFN 子层？
- 最近的 "deep and thin vs shallow and wide" 架构选择讨论了什么？
