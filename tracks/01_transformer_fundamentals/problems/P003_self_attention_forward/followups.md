# 面试追问

## 基础理解
- Self-attention 和 cross-attention 的区别是什么？Q/K/V 分别来自哪里？
- 为什么 Q、K、V 需要三个独立的投影矩阵？能不能共享？
- 不加 mask 的 self-attention 等价于什么操作？（全连接？）

## 和其他组件的关系
- P003 相比 P001 多了什么？相比 P002 少了什么？
- 在完整的 Transformer Block 中，self-attention 之后通常接什么？
- 如何从 P003 扩展到 P002（加多头支持）？需要改哪些代码？

## 高级话题
- Self-attention 的 O(n²) 瓶颈对长序列有什么影响？
- 线性 attention（如 Linear Transformer）如何把复杂度降到 O(n)？代价是什么？
- Self-attention 中每个 token 都参与 QKV 投影，这和 RNN 的对比如何？
