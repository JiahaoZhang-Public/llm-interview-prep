# 知识卡片

## 为什么 attention score 要除以 sqrt(d_k)？

为了防止点积过大导致 softmax 饱和，从而稳定梯度。

## Residual path 的核心作用是什么？

保留恒等映射路径，让深层网络更容易优化。

## RoPE 为什么适合 decoder 推理？

它把相对位置信息直接编码到 Q/K 中，不需要额外位置表。

## 为什么 decoder-only LM 必须使用 causal mask？

否则训练和推理都会看到未来 token，目标被泄漏。
