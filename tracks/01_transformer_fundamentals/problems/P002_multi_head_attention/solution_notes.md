# P002 Multi-Head Attention 题解

## 核心思路

1. 用三个线性层把输入投影到 Q、K、V
2. 把 hidden_size 拆成 num_heads × head_dim，reshape + transpose 得到 (B, H, T, D)
3. 每个 head 独立做 scaled dot-product attention
4. concat 所有 head 的输出，再过一个 output projection

关键：多头让模型能同时关注不同位置的不同表示子空间。

## 面试口述模板

> "Multi-head attention 的核心是用多个独立的 attention head 去捕获不同的表示子空间。
> 实现上就是把 hidden_size 拆成 H 个 head，每个 head 维度 d_k = hidden_size / H。
> 先分别用 W_Q、W_K、W_V 投影，再 reshape 成 (B, H, T, d_k)，
> 每个 head 独立做 scaled dot-product attention，
> 最后 concat 回 (B, T, hidden_size)，过一层 W_O 输出。"

## 常见坑

- **head_dim 计算不对**：hidden_size 必须能被 num_heads 整除，否则拆分会出错
- **reshape/transpose 顺序**：先 view 成 (B,T,H,D) 再 transpose(1,2) 变成 (B,H,T,D)
- **concat 后忘记 contiguous()**：transpose 后 memory layout 不连续，view 前要 contiguous
- **output projection 漏掉**：concat 后必须过 W_O，这是不同 head 信息融合的关键

## 复杂度

- 时间：O(n² · d)，和单头相同（只是拆成多个小头并行）
- 空间：O(H · n²) 存各 head 的 attention weights
- 参数量：3 × d² + d²（QKV 投影 + output 投影）
