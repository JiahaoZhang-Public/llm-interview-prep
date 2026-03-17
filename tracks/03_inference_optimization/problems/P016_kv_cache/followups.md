# 面试追问

## 基础理解
- 为什么没有 KV Cache 时自回归生成是 O(n^3)？加了之后变成 O(n^2)？
- KV Cache 只缓存 K 和 V，为什么不缓存 Q？
- 多头 attention 的 KV Cache shape 应该是什么？画出维度。

## 工程实现
- 反复 `torch.cat` 拼接 Tensor 有什么性能问题？如何用预分配 buffer 优化？
- 每层 Transformer 都要独立的 KV Cache 吗？请估算 LLaMA-70B 在 4096 seq_len 下的 KV Cache 显存。
- 请求结束后如何 reset？多并发请求如何管理各自的 cache？

## 高级话题
- 什么是 Paged Attention？它如何解决 KV Cache 的显存碎片化问题？
- GQA (Grouped Query Attention) 如何影响 KV Cache 的大小？
- 什么是 Sliding Window Attention？它对 KV Cache 有什么影响？
- 解释 prefix caching 如何复用不同请求之间的 KV Cache。
