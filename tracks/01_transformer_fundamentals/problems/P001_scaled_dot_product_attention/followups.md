# 面试追问

## 基础理解
- 为什么要除以 √d_k？如果不做 scaling 会怎样？从方差角度推导。
- softmax 数值不稳定时怎么处理？（提示：减去 max）PyTorch 内部做了吗？
- mask 用加法（-inf）而不是乘法（0）的原因是什么？

## 变体与扩展
- Self-attention vs Cross-attention：输入的 Q/K/V 来源有什么不同？
- 如果改成 cross-attention（Q 来自 decoder，K/V 来自 encoder），代码需要改什么？
- mask 在 batch 维和 head 维上如何 broadcast？

## 高级话题
- FlashAttention 如何避免 O(n²) 的中间存储？它的核心 trick 是什么？
- Multi-Query Attention (MQA) 和 Grouped-Query Attention (GQA) 对 attention 计算有什么影响？
- 为什么某些模型用 cosine attention 或 linear attention 替代 softmax attention？
