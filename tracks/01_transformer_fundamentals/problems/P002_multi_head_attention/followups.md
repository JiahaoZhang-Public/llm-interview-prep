# 面试追问

## 基础理解
- hidden_size 不能整除 num_heads 时怎么办？为什么要求整除？
- 为什么 reshape 后要 transpose(1,2)？直接 view 成 (B,H,T,D) 行不行？
- 为什么 concat 前需要 contiguous()？什么时候 tensor 会不连续？

## 变体与优化
- Multi-Query Attention (MQA)：K/V 只有一个 head，有什么好处和坏处？
- Grouped-Query Attention (GQA)：K/V 的 head 数是 Q 的 1/G，如何实现？
- QKV 三组投影矩阵可以融合成一个 (3*hidden, hidden) 的矩阵吗？好处是什么？

## 工程问题
- dropout 通常放在 attention weights 之后还是 output projection 之后？
- 如何在多 GPU 上做 tensor parallelism 切分 attention？
- FlashAttention 如何处理多头的计算？
